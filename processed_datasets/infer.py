import os
import sys
import numpy as np
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dozer_I:
    """
    Inference wrapper for Dozerformer — mirrors DAN's Inference.py (DAN_I) interface.

    Usage:
        sys.path.insert(0, "<path_to_Dozerformer_project>")
        from models.Inference_Dozer import Dozer_I
        from models.my_method import dozerformer   # imported inside after sys.path set

        inf = Dozer_I(args, dozerformer_model_cls=dozerformer.Model)
        inf.model_load(
            checkpoint_path="./checkpoints/<setting>/checkpoint.pth",
            norm_path="./checkpoints/<setting>/Norm.txt",
        )
        pred, gt = inf.test_single(
            test_point="2022-01-01 00:30:00",
            stream_csv_path="./data_provider/datasets/Ross_S_fixed.csv",
        )

    Norm.txt format (two values, one per line):
        <mean>
        <std>
    Save after training:
        np.savetxt("Norm.txt", [scale_stat["mean"], scale_stat["std"]])
    """

    def __init__(self, args, dozerformer_model_cls):
        """
        Parameters
        ----------
        args                  : argparse.Namespace — same args used during training
        dozerformer_model_cls : the Model class from dozerformer.py
                                (pass dozerformer.Model after adding Dozerformer to sys.path)
        """
        self.args = args
        self.in_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        self.model = dozerformer_model_cls(args).float().to(device)
        self.mean = None
        self.std = None

    # ------------------------------------------------------------------
    # Model + norm loading
    # ------------------------------------------------------------------

    def model_load(self, checkpoint_path, norm_path):
        """
        Load model weights and log+z-score normalisation statistics.

        norm_path : path to Norm.txt with two values: mean (line 1), std (line 2)
        """
        norm = np.loadtxt(norm_path, dtype=float)
        self.mean = float(norm[0])
        self.std = float(norm[1])
        print(f"Norm  mean={self.mean:.4f}  std={self.std:.4f}")

        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Data loading  (mirrors DAN_I.get_data)
    # ------------------------------------------------------------------

    def get_data(self, test_point, stream_csv_path):
        """
        Read in_len steps of stream data ending at test_point, and
        pred_len ground-truth steps immediately after.

        Returns
        -------
        stream_data : list of float, length in_len
        gt          : list of float, length pred_len
        """
        df = pd.read_csv(stream_csv_path, sep="\t")
        df.columns = ["id", "datetime", "value"]
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        point = df[df["datetime"] == test_point].index.values[0]
        stream_data = df[point - self.in_len: point]["value"].values.tolist()
        gt = df[point: point + self.pred_len]["value"].values.tolist()

        if np.isnan(stream_data).any():
            print("Warning: NaN in stream input sequence.")
        if np.isnan(gt).any():
            print("Warning: NaN in ground truth sequence.")

        return stream_data, gt

    # ------------------------------------------------------------------
    # Prediction  (mirrors DAN_I.predict)
    # ------------------------------------------------------------------

    def predict(self, stream_data):
        """
        Run one forward pass and return predictions in original scale.

        Parameters
        ----------
        stream_data : array-like, length in_len  (raw, un-normalised values)

        Returns
        -------
        pred : np.ndarray, shape (pred_len,)  in original streamflow units
        """
        # 1. Log + z-score normalisation  (matches DS_Dozer.py / DS.py)
        x = np.array(stream_data, dtype=np.float32)
        x_norm = (np.log(x + 1) - self.mean) / self.std  # (in_len,)

        # 2. Build encoder input  (1, in_len, 1)
        x_enc = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(-1).to(device)

        # 3. Build decoder input  (1, label_len + pred_len, 1)
        #    label_len context from tail of encoder, then zeros for the forecast
        dec_zeros = torch.zeros(1, self.pred_len, 1, dtype=torch.float32).to(device)
        if self.label_len > 0:
            label_part = x_enc[:, -self.label_len:, :]        # (1, label_len, 1)
            x_dec = torch.cat([label_part, dec_zeros], dim=1)  # (1, label+pred, 1)
        else:
            x_dec = dec_zeros                                  # (1, pred_len, 1)

        # 4. Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_enc, x_dec)  # (1, pred_len, 1)

        pred_norm = output[0, :, 0].cpu().numpy()  # (pred_len,)

        # 5. Inverse transform: undo z-score then log  (mirrors log_std_denorm_dataset)
        pred = np.exp(pred_norm * self.std + self.mean) - 1

        # 6. ReLU clip — no negative streamflow  (mirrors DAN_I.predict)
        pred = (pred + np.abs(pred)) / 2

        return pred

    # ------------------------------------------------------------------
    # Convenience wrapper  (mirrors DAN_I.test_single)
    # ------------------------------------------------------------------

    def test_single(self, test_point, stream_csv_path):
        """
        Load data for test_point, run prediction, return (pred, gt).

        Parameters
        ----------
        test_point      : str  e.g. "2022-01-01 00:30:00"
        stream_csv_path : str  path to the tab-separated stream CSV

        Returns
        -------
        pred : np.ndarray, shape (pred_len,)
        gt   : list of float, length pred_len
        """
        stream_data, gt = self.get_data(test_point, stream_csv_path)
        pred = self.predict(stream_data)
        return pred, gt

    # ------------------------------------------------------------------
    # Full test-set evaluation  (mirrors DS_Dozer.py gen_test_data logic)
    # ------------------------------------------------------------------

    def test_all(
        self,
        stream_csv_path,
        test_start,
        test_end,
        test_stride=16,
    ):
        """
        Run inference on every test window exactly as DS_Dozer builds the test set:
          - Data slice: [test_start - in_len, test_end]
          - Windows: s_begin = i * test_stride, num_windows = int((N - in_len - pred_len) / test_stride)
          - NaN check: skip windows with any NaN in the full in_len+pred_len span
          - No hydro-year filter (matches DAN gen_test_data)

        Parameters
        ----------
        stream_csv_path : str   path to tab-separated stream CSV
        test_start      : str   e.g. "2021-09-01 00:30:00"
        test_end        : str   e.g. "2022-05-31 23:30:00"
        test_stride     : int   step between windows (default 16, every 4 hours at 15-min data)

        Returns
        -------
        preds : np.ndarray, shape (num_windows, pred_len)  — original scale
        trues : np.ndarray, shape (num_windows, pred_len)  — original scale
        """
        # --- Load and slice raw data ---
        df = pd.read_csv(stream_csv_path, sep="\t")
        df.columns = ["id", "datetime", "value"]
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        test_start_idx = df[df["datetime"] == test_start].index[0]
        test_end_idx   = df[df["datetime"] == test_end].index[0]

        b1 = test_start_idx - self.in_len
        b2 = test_end_idx

        raw = df["value"].values.astype(np.float32)[b1: b2 + 1]

        # --- Log + z-score normalisation ---
        data = (np.log(raw + 1) - self.mean) / self.std

        # --- Build window indices (matches DS_Dozer test logic exactly) ---
        N = len(data)
        num_windows = int((N - self.in_len - self.pred_len) / test_stride)

        preds, trues = [], []

        self.model.eval()
        with torch.no_grad():
            for i in range(num_windows):
                s_begin = i * test_stride
                s_end   = s_begin + self.in_len
                t_end   = s_end + self.pred_len

                # Skip NaN windows
                if np.isnan(data[s_begin: t_end]).any():
                    continue

                x_norm = data[s_begin: s_end].astype(np.float32)   # (in_len,)
                y_norm = data[s_end: t_end].astype(np.float32)      # (pred_len,)

                # Encoder input  (1, in_len, 1)
                x_enc = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(-1).to(device)

                # Decoder input  (1, label_len + pred_len, 1)
                dec_zeros = torch.zeros(1, self.pred_len, 1, dtype=torch.float32).to(device)
                if self.label_len > 0:
                    label_part = x_enc[:, -self.label_len:, :]
                    x_dec = torch.cat([label_part, dec_zeros], dim=1)
                else:
                    x_dec = dec_zeros

                output = self.model(x_enc, x_dec)           # (1, pred_len, 1)
                pred_norm = output[0, :, 0].cpu().numpy()   # (pred_len,)

                # Inverse transform
                pred = np.exp(pred_norm * self.std + self.mean) - 1
                pred = (pred + np.abs(pred)) / 2            # ReLU clip

                true = np.exp(y_norm * self.std + self.mean) - 1

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)  # (num_windows, pred_len)
        trues = np.array(trues)  # (num_windows, pred_len)

        print(f"Test windows evaluated: {len(preds)}")
        return preds, trues
