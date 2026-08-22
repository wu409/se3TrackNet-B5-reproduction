import os
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from b5_policy import se3_log_map, se3_exp_map, compute_se3_prior, init_b5_state, b5_transition
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
from scipy.optimize import minimize
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import argparse
from scipy.stats import t
import hashlib
import pandas as pd


K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)

# ==================== 1. SE(3) 李群辅助函数 ====================
def calc_auc(errors_cm, max_threshold_cm=10.0):
    errors_array = np.array(errors_cm)
    if len(errors_array) == 0: return 0.0
    thresholds = np.linspace(0, max_threshold_cm, 1000)
    accs = [np.mean(errors_array <= th) for th in thresholds]
    return (np.trapz(accs, thresholds) / max_threshold_cm) * 100.0

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(labels[in_bin]) - np.mean(probs[in_bin])) * (np.sum(in_bin) / len(probs))
    return ece

def compute_episode_level_ci(scores, n_bootstraps=2000, confidence_level=0.95):
    """按 Episode 计算 95% 置信区间 """
    scores_arr = np.array(scores)
    if len(scores_arr) < 2:
        m = np.mean(scores_arr)
        return m, m, m # 单 Episode 直接返回均值

    bootstrapped_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1.0 - alpha) * 100)
    mean_score = np.mean(scores_arr)

    return mean_score, lower_bound, upper_bound



def compute_paired_bootstrap_ci(differences, n_bootstraps=10000, confidence_level=0.95, seed=42):
    """对 episode-level paired differences 做 bootstrap CI。"""
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    mean_diff = float(np.mean(diffs))
    if len(diffs) == 1:
        return mean_diff, mean_diff, mean_diff
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means[b] = np.mean(sample)
    alpha = (1.0 - confidence_level) / 2.0
    return mean_diff, float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100))

def compute_full_sha256(filepath):
    """计算文件的完整 64 位 SHA-256 哈希值 (读全量数据，绝不截断)"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def resolve_manifest_path(path_value, manifest_path):
    """兼容 manifest 中的绝对路径或相对路径。"""
    path_value = str(path_value)
    if os.path.isabs(path_value) or os.path.exists(path_value):
        return path_value
    candidate = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path_value)
    return candidate

def load_episode_manifest(manifest_path, sequence, data_dir, gt_dir, result_dir):
    """
    从冻结的 reference manifest 中按 exact sequence 取出一个 episode。
    本函数只消费 reference，并再次检查实际 path + SHA-256；不会重建映射。
    """
    df_manifest = pd.read_csv(manifest_path)
    required = {
        "sequence", "sequence_index", "frame_id", "depth_path", "gt_path", "pred_path",
        "depth_sha256", "gt_sha256", "pred_sha256",
        "association_method", "association_reference", "association_description",
    }
    missing = required - set(df_manifest.columns)
    if missing:
        raise ValueError(f"Master manifest 缺少字段: {sorted(missing)}")
    episode = df_manifest[df_manifest["sequence"] == sequence].copy()
    if len(episode) == 0:
        raise ValueError(f"Master manifest 中找不到 episode: {sequence}")
    episode["frame_id"] = episode["frame_id"].astype(int)
    episode["sequence_index"] = episode["sequence_index"].astype(int)
    if episode["frame_id"].duplicated().any():
        dup = episode.loc[episode["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"[{sequence}] manifest 存在重复 frame_id: {dup}")
    if episode["sequence_index"].duplicated().any():
        dup = episode.loc[episode["sequence_index"].duplicated(), "sequence_index"].tolist()
        raise ValueError(f"[{sequence}] manifest 存在重复 sequence_index: {dup}")
    episode = episode.sort_values("sequence_index", kind="stable").reset_index(drop=True)
    expected_indices = list(range(len(episode)))
    if episode["sequence_index"].tolist() != expected_indices:
        raise ValueError(
            f"[{sequence}] manifest sequence_index不连续: "
            f"expected={expected_indices[:10]}, actual={episode['sequence_index'].tolist()[:10]}"
        )
    methods = episode["association_method"].dropna().unique().tolist()
    if methods != ["official_ycbineoat_reference_sorted_index"]:
        raise ValueError(f"[{sequence}] manifest association_method不受支持: {methods}")
    episode["seq_idx"] = episode["sequence_index"]
    episode["frame_idx"] = episode["frame_id"]

    episode["depth_path"] = episode["depth_path"].map(lambda x: os.path.join(data_dir, x))
    episode["gt_path"] = episode["gt_path"].map(lambda x: os.path.join(gt_dir, os.path.basename(x)))
    episode["pred_path"] = episode["pred_path"].map(lambda x: os.path.join(result_dir, os.path.basename(x)))

    for col in ["depth_path", "gt_path", "pred_path"]:
        missing_paths = [p for p in episode[col].tolist() if not os.path.exists(p)]
        if missing_paths:
            raise FileNotFoundError(f"[{sequence}] {col} 中存在不存在的文件，例如: {missing_paths[0]}")
    for row in episode.itertuples(index=False):
        for artifact_type in ["depth", "gt", "pred"]:
            path = getattr(row, f"{artifact_type}_path")
            expected = str(getattr(row, f"{artifact_type}_sha256")).lower()
            actual = compute_full_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"[{sequence} frame {row.frame_id}] {artifact_type} SHA-256 mismatch: "
                    f"expected={expected}, actual={actual}, path={path}"
                )
    print(f"[{sequence}] manifest paths and SHA-256 values verified: {len(episode)} frames")
    return episode

def main(args):
    np.random.seed(args.seed)
    try:
        o3d.utility.random.seed(args.seed)
    except Exception:
        pass
    # ==================== 2. 载入数据集与训练好的风险分类器 ====================
    print("正在载入 CSV 数据集并训练分类器...")
    df = pd.read_csv(args.csv_path)
    required_label_columns = {"sequence", "frame_id", "obs_risk_label", "prior_risk_label"}
    missing_label_columns = required_label_columns - set(df.columns)
    if missing_label_columns:
        raise ValueError(f"Label CSV 缺少字段: {sorted(missing_label_columns)}")
    if df.duplicated(["sequence", "frame_id"]).any():
        duplicated = df.loc[df.duplicated(["sequence", "frame_id"], keep=False), ["sequence", "frame_id"]]
        raise ValueError(f"Label CSV 存在重复(sequence, frame_id): {duplicated.head(10).to_dict('records')}")
    path = args.result_dir
    test_seq = os.path.basename(path)
    train_pattern = "|".join(args.train_seqs)
    train_df = df[df['sequence'].str.contains(train_pattern)]

    matched_train_seqs = df[df['sequence'].str.contains(train_pattern)]['sequence'].unique()  


    #对匹配到的每一个训练序列，按时间轴切成前 80% (训练) 和 后 20% (校准)
    train_dfs, cal_dfs = [], []
    for seq in matched_train_seqs:
        sub_df = df[df['sequence'] == seq]
        split_idx = int(len(sub_df) * 0.7) # 70% 时间截断点
        
        train_dfs.append(sub_df.iloc[:split_idx]) # 前 70% 时间段进入训练集
        cal_dfs.append(sub_df.iloc[split_idx:])   # 后 30% 时间段进入校准集

    train_df = pd.concat(train_dfs)
    cal_df   = pd.concat(cal_dfs)

    test_df = df[df['sequence'] == test_seq].copy()
    test_df['frame_id'] = test_df['frame_id'].astype(int)
    if test_df['frame_id'].duplicated().any():
        raise ValueError(f"测试序列 {test_seq} 存在重复 frame_id，拒绝构建概率字典")
    test_df = test_df.set_index('frame_id')

    test_ALL_df = df[df['sequence'].str.contains(args.test_base_seq)]
    
    print(f"训练集包含序列关键词: {args.train_seqs} | 行数: {len(train_df)}")
    print(f"测试集 (单序列) 名称: {test_seq} | 行数: {len(test_df)}")
    print(f"测试集 (全量变体) 名称: {args.test_base_seq} | 行数: {len(test_ALL_df)}")

    if len(test_df) == 0:
        print("错误: 测试集为空，请检查 CSV 文件中的序列名称！")

    feature_cols = ['x1_depth_residual', 'x2_inlier_ratio','x3_innovation_mag','x4_support_ratio']

    X_train = train_df[feature_cols].values
    X_cal = cal_df[feature_cols].values
    X_test = test_df[feature_cols].values
    X_ALL_test = test_ALL_df[feature_cols].values

    y_obs_train = train_df['obs_risk_label'].values
    y_obs_cal = cal_df['obs_risk_label'].values
    y_obs_test = test_df['obs_risk_label'].values
    y_obs_ALL_test = test_ALL_df['obs_risk_label'].values

    y_prior_train = train_df['prior_risk_label'].values
    y_prior_cal = cal_df['prior_risk_label'].values
    y_prior_test = test_df['prior_risk_label'].values
    y_prior_ALL_test = test_ALL_df['prior_risk_label'].values

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_cal_scaled = scaler_x.transform(X_cal)
    X_test_scaled = scaler_x.transform(X_test)
    X_test_ALL_scaled = scaler_x.transform(X_ALL_test)

    clf_obs = LogisticRegression(max_iter=1000)
    clf_obs.fit(X_train_scaled, y_obs_train)

    clf_prior = LogisticRegression(max_iter=1000)
    clf_prior.fit(X_train_scaled,y_prior_train)

    # 4. 温度缩放标定 (Temperature Scaling，保证概率不盲目自信)
    cal_logits_obs = clf_obs.decision_function(X_cal_scaled)
    test_logits_obs = clf_obs.decision_function(X_test_scaled)
    test_ALL_logits_obs = clf_obs.decision_function(X_test_ALL_scaled)

    cal_logits_prior = clf_prior.decision_function(X_cal_scaled)
    test_logits_prior = clf_prior.decision_function(X_test_scaled)
    test_ALL_logits_prior = clf_prior.decision_function(X_test_ALL_scaled)


    def eval_loss_obs(t):
        scaled = cal_logits_obs / t[0]
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(y_obs_cal * np.log(probs) + (1 - y_obs_cal) * np.log(1 - probs))
    
    def eval_loss_prior(t):
            scaled = cal_logits_prior / t[0]
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y_prior_cal * np.log(probs) + (1 - y_prior_cal) * np.log(1 - probs))

    res_obs = minimize(eval_loss_obs, [1.0], bounds=[(0.01, 10.0)])
    temp_factor_obs = res_obs.x[0]

    res_prior = minimize(eval_loss_prior, [1.0], bounds=[(0.01, 10.0)])
    temp_factor_prior = res_prior.x[0]

    p_obs_bad = 1.0 / (1.0 + np.exp(-(test_logits_obs / temp_factor_obs)))
    p_obs_bad_ALL = 1.0 / (1.0 + np.exp(-(test_ALL_logits_obs / temp_factor_obs)))

    p_prior_bad = 1.0 / (1.0 + np.exp(-(test_logits_prior / temp_factor_prior)))
    p_prior_bad_ALL = 1.0 / (1.0 + np.exp(-(test_ALL_logits_prior / temp_factor_prior)))

    p_obs_bad_dict = dict(zip(test_df.index.tolist(),p_obs_bad.tolist()))
    p_prior_bad_dict = dict(zip(test_df.index.tolist(),p_prior_bad.tolist()))
    support_dict = test_df['x4_support_ratio'].to_dict()

    for fid in list(test_df.index[:10]):
        print(
            f"frame_id={fid:4d} | "
            f"P(obs_bad)={p_obs_bad_dict[fid]:.4f} | "
            f"P(prior_bad)={p_prior_bad_dict[fid]:.4f} | "
            f"support={support_dict[fid]:.4f}"
        )
    
    # ==================== 3. 计算 4 大概率与标定指标 ====================
    auroc_obs = roc_auc_score(y_obs_ALL_test, p_obs_bad_ALL)
    precision_obs, recall_obs, _ = precision_recall_curve(y_obs_ALL_test, p_obs_bad_ALL)
    auprc_obs = auc(recall_obs, precision_obs)
    brier_obs = brier_score_loss(y_obs_ALL_test, p_obs_bad_ALL)
    ece_obs = compute_ece(p_obs_bad_ALL,y_obs_ALL_test)

    auroc_prior = roc_auc_score(y_prior_ALL_test, p_prior_bad_ALL)
    precision_prior, recall_prior, _ = precision_recall_curve(y_prior_ALL_test, p_prior_bad_ALL)
    auprc_prior = auc(recall_prior, precision_prior)
    brier_prior = brier_score_loss(y_prior_ALL_test, p_prior_bad_ALL)
    ece_prior = compute_ece(p_prior_bad_ALL, y_prior_ALL_test)


    # ==================== 5. 运行 6 个 Baselines====================
    print("正在测试序列上运行 6 个 Baselines PK...")

    
    point_path = args.point_path
    with open(point_path, 'r') as f:
        model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
    open3d_model = U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64))


    last_name = os.path.basename(args.result_dir)
    df_manifest = load_episode_manifest(args.manifest_path, last_name,  args.data_dir,args.gt_dir,args.result_dir)
    manifest_frame_ids = set(df_manifest["frame_id"].astype(int).tolist())
    risk_frame_ids = set(test_df.index.astype(int).tolist())
    if manifest_frame_ids != risk_frame_ids:
        missing_in_risk = sorted(manifest_frame_ids - risk_frame_ids)
        missing_in_manifest = sorted(risk_frame_ids - manifest_frame_ids)
        raise ValueError(
            f"[{last_name}] label CSV 与 master manifest 的 frame_id 不一致。"
            f" manifest-only={missing_in_risk[:10]}, label-only={missing_in_manifest[:10]}"
        )

    # 定义 6 个 Baselines 最终的逐帧姿态误差结果 (cm)
    b1_errs,  b2_errs,  b3_errs,  b4_errs,  b5_errs,  b6_errs  = [], [], [], [], [], []
    b5_modes = [] # 记录三模式历史
    matched_frames=[]
    false_recovery_triggers = 0
    recovery_latencies1,recovery_latencies2,recovery_latencies3,recovery_latencies4,recovery_latencies5,recovery_latencies6 = [],[],[],[],[],[] # 记录恢复延迟
    T_history2,T_history3,T_history4,T_history5,T_history6 =  [], [], [], [], []
    b5_state = init_b5_state()
    recovery_record = None


    #for i, frame_id in enumerate(matched_frames):
    for row in df_manifest.itertuples():
    # 直接从权威 Manifest 行中提取数据，绝对不可能错位！
        i = row.seq_idx
        frame_id = row.frame_idx
        matched_frames.append(frame_id)
        depth_file = row.depth_path
        gt_file    = row.gt_path
        pred_file  = row.pred_path

        # 读取姿态与深度图
        T_obs = np.loadtxt(pred_file).reshape(4, 4)
        T_gt  = np.loadtxt(gt_file).reshape(4, 4)
        depth_raw = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        
        if i < 2: 
            T_prior_current2,T_prior_current3,T_prior_current4,T_prior_current5,T_prior_current6 = T_obs,T_obs,T_obs,T_obs,T_obs  # 初始化阶段
        else:
            T_prior_current2 = compute_se3_prior(T_history2[-1],T_history2[-2])
            T_prior_current3 = compute_se3_prior(T_history3[-1],T_history3[-2])
            T_prior_current4 = compute_se3_prior(T_history4[-1],T_history4[-2])
            T_prior_current5 = compute_se3_prior(T_history5[-1],T_history5[-2])
            T_prior_current6 = compute_se3_prior(T_history6[-1],T_history6[-2])

        p_obs_bad = p_obs_bad_dict[frame_id]
        p_prior_bad = p_prior_bad_dict[frame_id]
        support = support_dict[frame_id]
        # B1: Obs-Only 
        #b1_errs.append(e_obs)
        b1_errs.append(U.adi(T_obs,T_gt,open3d_model)* 100)
        
        # B2: Fixed-alpha Smoothing (0.5)
        delta=se3_log_map(np.linalg.inv(T_prior_current2)@T_obs)
        T_final2=T_prior_current2 @ se3_exp_map(args.alpha*delta)
        b2_errs.append(U.adi(T_final2,T_gt,open3d_model)* 100)
        T_history2.append(T_final2)

        # B3: Hard Depth Threshold (深度缺失则听惯性)
        if support<0.4:
            T_final3 = T_obs
        else:
            T_final3 = T_prior_current3
        T_history3.append(T_final3)
        b3_errs.append(U.adi(T_final3,T_gt,open3d_model)* 100)

        # B4: Robust Huber Weighting
        innovation=se3_log_map(np.linalg.inv(T_prior_current4)@T_obs)
        r=np.linalg.norm(innovation)
        delta=0.1
        if r<=delta:
            alpha=1.0
        else:
            alpha=delta/r
        T_huber=T_prior_current4@se3_exp_map(alpha*innovation)
        T_history4.append(T_huber)
        b4_errs.append(U.adi(T_huber,T_gt,open3d_model)* 100)

        #  B5: Proposed Three-Mode Policy
        depth_real = depth_raw.astype(np.float32) / 1000.0
        T_final, current_mode, b5_state, recovery_info = b5_transition(
            T_obs=T_obs,
            T_prior=T_prior_current5,
            p_obs_bad=p_obs_bad,
            p_prior_bad=p_prior_bad,
            support=support,
            depth_real=depth_real,
            model_pts=model_pts,
            K=K,
            p_risk_threshold=args.p_risk_threshold,
            frame_index=i,
            frame_id=frame_id,
            state=b5_state,
            blackout_min_frames=args.blackout_min_frames,
            use_prior_predictor=True
        )

        T_history5.append(T_final)
        b5_error_current = U.adi(T_final, T_gt, open3d_model) * 100
        b5_errs.append(b5_error_current)
        b5_modes.append(current_mode)

        # 这里只记录“independent recovery action 本身”的误差；若 recovery action 失败，则记 NaN。
        if recovery_info is not None and recovery_record is None:
            if recovery_info["recovery_success"]:
                b5_recovery_error_cm = U.adi(
                    recovery_info["T_recovery"], T_gt, open3d_model
                ) * 100
            else:
                b5_recovery_error_cm = np.nan
            b1_recovery_error_cm = b1_errs[-1]
            recovery_record = {
                "episode": last_name,
                "recovery_frame": int(frame_id),
                "recovery_success": bool(recovery_info["recovery_success"]),
                "B1_error_cm": float(b1_recovery_error_cm),
                "B5_recovery_error_cm": float(b5_recovery_error_cm) if np.isfinite(b5_recovery_error_cm) else np.nan,
                "B5_operational_error_cm": float(b5_error_current),
                "B5_minus_B1_cm": float(b5_recovery_error_cm - b1_recovery_error_cm) if np.isfinite(b5_recovery_error_cm) else np.nan
            }

        # B6: Oracle Decision Policy
        err_obs = U.adi(T_obs, T_gt,open3d_model)
        err_prior = U.adi(T_prior_current6,T_gt,open3d_model)
        if err_obs < err_prior:
            T_final_oracle = T_obs
        else:
            T_final_oracle = T_prior_current6
        T_history6.append(T_final_oracle)
        b6_errs.append(U.adi(T_final_oracle,T_gt,open3d_model)*100)

        # T_final2=T_prior_current2
        # T_history2.append(T_final2)
        #b7.append(U.adi(T_final2,T_gts[i],open3d_model)*100)
       
    blackout_start = b5_state["blackout_start_idx"]
    blackout_end = b5_state["blackout_end_idx"]
    print(Counter(b5_modes))
    print("black_start:", blackout_start, "| frame:", b5_state["blackout_start_frame"])
    print("black_end:", b5_state["last_blackout_idx"], "| frame:", b5_state["blackout_end_frame"])
    print("recovery_index:", blackout_end, "| frame:", b5_state["recovery_frame"])
    # print(b5_errs[42:200])

    # ==================== 动作 A: 测量 【恢复延迟 Recovery Latency】 ====================
    # 找到黑屏结束点 (第 160 帧)，测量复活需要几帧
    a=0.5
    if "black" in args.result_dir:
        print("computing recovery_latencies")
        for i in range(blackout_end, len(b5_errs)):
            if b1_errs[i] < a: 
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies1.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b2_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies2.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b3_errs[i] < a: 

                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies3.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b4_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies4.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b5_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies5.append(latency)
                break
        for i in range(blackout_end, len(b5_errs)):
            if b6_errs[i] < a: 
                
                latency = i - blackout_end  # 恢复延迟帧数 (例如 1 帧)
                recovery_latencies6.append(latency)
                break


    if len(recovery_latencies5) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency5  = f"{recovery_latencies5[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency5  = "N/A (Failed)"

    if len(recovery_latencies1) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency1  = f"{recovery_latencies1[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency1  = "N/A (Failed)"

    if len(recovery_latencies2) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency2  = f"{recovery_latencies2[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency2  = "N/A (Failed)"

    if len(recovery_latencies3) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency3  = f"{recovery_latencies3[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency3  = "N/A (Failed)"

    if len(recovery_latencies4) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency4  = f"{recovery_latencies4[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency4  = "N/A (Failed)"

    if len(recovery_latencies6) > 0:
        # 恢复成功：记录真实的复活帧数
        avg_recovery_latency6  = f"{recovery_latencies6[0]:.1f} frames"
    else:
        # 恢复失败 (黑屏后彻底跟丢)：记录为 N/A (Failed)！
        avg_recovery_latency6  = "N/A (Failed)"

    # ==================== 动作 C: 保存逐帧 CSV 日志与黑屏复活轨迹图 ====================
    # ==================== 6. 保存图像与 CSV 日志 ====================
    # 保存逐帧 CSV 日志
    df_log = pd.DataFrame({
        "frame_id": matched_frames,
        "p_obs_bad": [p_obs_bad_dict[f] for f in matched_frames],
        "p_prior_bad": [p_prior_bad_dict[f] for f in matched_frames],
        "selected_mode": b5_modes,
        "error_b1_obs_cm": b1_errs,
        "error_b2_obs_cm": b2_errs,
        "error_b3_obs_cm": b3_errs,
        "error_b4_obs_cm": b4_errs,
        "error_b5_ours_cm": b5_errs,
        "error_b6_obs_cm": b6_errs,

    })
    df_log.to_csv(f"./checkpoint2_per_frame_{last_name}_log_threshold{args.risk_threshold}.csv", index=False)

    # 保存 Reliability Diagram
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (p_obs_bad_ALL > bin_boundaries[i]) & (p_obs_bad_ALL <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_obs_ALL_test[in_bin]))
            bin_confs.append(np.mean(p_obs_bad_ALL[in_bin]))
    plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Ours (ECE={ece_obs:.3f})')
    plt.title(r'Reliability Diagram for Observation Risk', fontsize=11)
    plt.xlabel(r'Predicted $P(\mathrm{Observation\ Bad})$', fontsize=11)
    plt.ylabel('Empirical Observed Help Frequency', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig(f'reliability_diagram_observation_risk_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0)')
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (p_prior_bad_ALL > bin_boundaries[i]) & (p_prior_bad_ALL <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_prior_ALL_test[in_bin]))
            bin_confs.append(np.mean(p_prior_bad_ALL[in_bin]))
    plt.plot(bin_confs, bin_accs, 's-', color='darkorange', linewidth=2, label=f'Ours (ECE={ece_prior:.3f})')
    plt.title(r'Reliability Diagram for Prior Risk', fontsize=11)
    plt.xlabel(r'Predicted $P(\mathrm{Prior\ Bad})$', fontsize=11)
    plt.ylabel('Empirical Observed Help Frequency', fontsize=11)
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig(f'reliability_diagram_prior_risk_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    if "black" in args.result_dir: 
        # 保存 Trajectory Trace Plot
        plt.figure(figsize=(10, 5))
        plt.plot(b1_errs[:], 'r-', label='B1: Obs-Only (Diverges on Blackout)', alpha=0.7)
        plt.plot(b5_errs[:], 'g--', label='B5: Ours (Three-Mode Policy)', linewidth=2)
        plt.axvspan(blackout_start, blackout_end, color='gray', alpha=0.3, label='10-Frame Complete Blackout')
        plt.title('Temporal Trajectory Trace: Pose Error & Recovery under Blackout', fontsize=12)
        plt.xlabel('Frame Number', fontsize=11)
        plt.ylabel('ADD-S Pose Error (cm)', fontsize=11)
        plt.legend(); plt.grid(True, linestyle='--')
        plt.savefig(f'trajectory_recovery_plot_threshold{args.risk_threshold}.png', dpi=300, bbox_inches='tight')

    print("\n所有的 11 项交付物已全部生成完毕！图片与 CSV 已成功保存！")


    adds_scores = [calc_auc(b1_errs), calc_auc(b2_errs), calc_auc(b3_errs), calc_auc(b4_errs), calc_auc(b5_errs), calc_auc(b6_errs)]
    fail_rates = [np.mean(np.array(b1_errs) > 2.0)*100, np.mean(np.array(b2_errs) > 2.0)*100,np.mean(np.array(b3_errs) > 2.0)*100, np.mean(np.array(b4_errs) > 2.0)*100,np.mean(np.array(b5_errs) > 2.0)*100, np.mean(np.array(b6_errs) > 2.0)*100]
    latency_scores = [avg_recovery_latency1, avg_recovery_latency2, avg_recovery_latency3, avg_recovery_latency4, avg_recovery_latency5, avg_recovery_latency6]
    false_triggers = ["N/A", "N/A", "N/A", "N/A", f"{false_recovery_triggers} times", "0 times"]
    prob_metrics_obs = {'auroc_obs': auroc_obs,'auprc_obs': auprc_obs,'brier_obs': brier_obs,'ece_obs': ece_obs,'temp_factor_obs': temp_factor_obs} 
    prob_metrics_prior = {'auroc_prior': auroc_prior,'auprc_prior': auprc_prior,'brier_prior': brier_prior,'ece_prior': ece_prior,'temp_factor_prior': temp_factor_prior} 
    

    blackout_intervals = []
    for interval in b5_state["blackout_intervals"]:
        interval_record = {"episode": last_name}
        interval_record.update(interval)
        blackout_intervals.append(interval_record)

    return (adds_scores, fail_rates, latency_scores, false_triggers, prob_metrics_obs,
            prob_metrics_prior, recovery_record, blackout_intervals)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="./per_frame_label_threshold1.0.csv", help="csv数据集路径")
    parser.add_argument('--manifest_path', type=str, default="./reference_manifest.csv",
                        help="冻结的 reference manifest（只读，不在 evaluation 中重建）")
    parser.add_argument('--result_dir', nargs='+',type=str, default=["./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4",
                                                                     "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5",], 
                                                                     help="要测试的所有序列路径")
    parser.add_argument('--gt_dir', type=str, default="./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses", help="GT_Pose Path")
    parser.add_argument('--point_path', type=str, default="./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz", help="point_path")
    parser.add_argument('--train_seqs', nargs='+', default=["bleach0", "mustard0"], help="训练集包含的序列关键字列表")
    parser.add_argument('--test_base_seq', type=str, default="bleach_hard_00_03_chaitanya", help="测试集物体的基础名称")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="受损数据集基础路径")
    parser.add_argument('--p_risk_threshold', type=float,  default=0.80, help="B5 risk probability threshold")
    parser.add_argument('--alpha', type=float, default=0.5, help="B2-alpha")
    parser.add_argument('--risk_threshold', type=float, default=1.0, help="risk_threshold")
    parser.add_argument('--blackout_min_frames', type=int, default=10, help="触发blackout recovery所需连续blackout帧数")
    parser.add_argument('--bootstrap_samples', type=int, default=10000, help="paired episode bootstrap次数")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    args = parser.parse_args()
    np.random.seed(args.seed)
    baseline_names = [
        "B1: Obs-Only se(3)-TrackNet",
        "B2: Fixed-Alpha (0.5) Interpolation",
        "B3: Hard Depth Threshold",
        "B4: Robust Huber Weighting",
        "B5: Proposed Three-Mode Policy (Ours)",
        "B6: Oracle Decision Policy (Upper Bound)"
    ]

    ADDS_dict = {}
    FAIL_dict = {}
    LATENCY_dict = {}
    TRIGGER_dict = {}
    RECOVERY_dict = {}
    BLACKOUT_INTERVALS_dict = {}
    
    last_prob_metrics_obs = None # 用于保存概率标定指标
    last_prob_metrics_prior = None

    result_dirs = args.result_dir 
    
    for single_dir in result_dirs:
        args.result_dir = single_dir 
        basename = os.path.basename(single_dir)
        
        print(f"\n正在评估 Episode: {basename} ...")
        
        # 接收 main(args) 返回的 5 个元组/列表！
        (adds_s, fails, lats, fts, prob_metrics_obs, prob_metrics_prior,
         recovery_record, blackout_intervals) = main(args)
        
        ADDS_dict[basename] = adds_s
        FAIL_dict[basename] = fails
        LATENCY_dict[basename] = lats
        TRIGGER_dict[basename] = fts
        RECOVERY_dict[basename] = recovery_record
        BLACKOUT_INTERVALS_dict[basename] = blackout_intervals
        last_prob_metrics_obs = prob_metrics_obs # 记录概率指标
        last_prob_metrics_prior = prob_metrics_prior 

    print("\n" + "="*100)
    print("="*100)

    summary_rows = []

    # 原始 episode summary 继续保留，但不再给每个 baseline 单独 bootstrap CI；
    for b_idx, b_name in enumerate(baseline_names):
        scores_for_this_baseline = [ADDS_dict[ep][b_idx] for ep in ADDS_dict.keys()]
        mean_score = float(np.mean(scores_for_this_baseline))
        row_data = {"Baseline / Method": b_name}
        for ep in ADDS_dict.keys():
            row_data[f"ADD-S ({ep})"] = f"{ADDS_dict[ep][b_idx]:.3f}%"
        row_data["Mean ADD-S (%)"] = f"{mean_score:.3f}%"
        for ep in FAIL_dict.keys():
            row_data[f"Failure Rate ({ep})"] = f"{FAIL_dict[ep][b_idx]:.3f}%"
        for ep in LATENCY_dict.keys():
            row_data[f"Latency ({ep})"] = LATENCY_dict[ep][b_idx]
        for ep in TRIGGER_dict.keys():
            row_data[f"False Triggers ({ep})"] = TRIGGER_dict[ep][b_idx]
        summary_rows.append(row_data)

    df_summary = pd.DataFrame(summary_rows)

    # ====================#4: paired episode-level AUC comparison ====================
    blackout_eps = [ep for ep in ADDS_dict.keys() if "black" in ep.lower()]
    if len(blackout_eps) != 5:
        print(f"paired AUC 预期 5 个 blackout episodes, 当前得到 {len(blackout_eps)} 个: {blackout_eps}")

    if blackout_eps:
        interval_rows = []
        for episode in blackout_eps:
            episode_intervals = BLACKOUT_INTERVALS_dict.get(episode, [])
            if len(episode_intervals) != 1:
                raise RuntimeError(
                    f"[{episode}] expected exactly one validated blackout interval, "
                    f"found {len(episode_intervals)}"
                )
            interval_rows.extend(episode_intervals)
        blackout_interval_path = f"./checkpoint2_blackout_frame_intervals_threshold{args.risk_threshold}.csv"
        pd.DataFrame(interval_rows).to_csv(blackout_interval_path, index=False)
        print(f"Exact blackout frame intervals saved: {blackout_interval_path}")

    paired_auc_rows = []
    auc_differences = []
    for ep in blackout_eps:
        auc_b1 = float(ADDS_dict[ep][0])
        auc_b5 = float(ADDS_dict[ep][4])
        delta_auc = auc_b5 - auc_b1
        auc_differences.append(delta_auc)
        paired_auc_rows.append({
            "episode": ep,
            "AUC_B1_percent": auc_b1,
            "AUC_B5_percent": auc_b5,
            "Delta_B5_minus_B1_percentage_points": delta_auc
        })

    auc_mean_diff, auc_ci_low, auc_ci_high = compute_paired_bootstrap_ci(
        auc_differences,
        n_bootstraps=args.bootstrap_samples,
        seed=args.seed
    )
    paired_auc_rows.append({
        "episode": "PAIRED_MEAN",
        "AUC_B1_percent": np.nan,
        "AUC_B5_percent": np.nan,
        "Delta_B5_minus_B1_percentage_points": auc_mean_diff,
        "Paired_95CI_low": auc_ci_low,
        "Paired_95CI_high": auc_ci_high
    })
    paired_auc_df = pd.DataFrame(paired_auc_rows)
    paired_auc_path = f"./checkpoint2_paired_auc_B5_vs_B1_threshold{args.risk_threshold}.csv"
    paired_auc_df.to_csv(paired_auc_path, index=False)
    print(f" Paired AUC comparison 已保存: {paired_auc_path}")
    print(f"   paired mean Δ(B5-B1) = {auc_mean_diff:.4f} percentage points")
    print(f"   paired episode-bootstrap 95% CI = [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

    # ====================  paired recovery-frame validation ====================
    recovery_rows = []
    recovery_differences = []
    for ep in blackout_eps:
        rec = RECOVERY_dict.get(ep)
        if rec is None:
            recovery_rows.append({
                "episode": ep,
                "recovery_frame": np.nan,
                "recovery_success": False,
                "B1_error_cm": np.nan,
                "B5_recovery_error_cm": np.nan,
                "B5_operational_error_cm": np.nan,
                "Delta_B5_minus_B1_cm": np.nan
            })
            continue
        recovery_rows.append({
            "episode": ep,
            "recovery_frame": rec["recovery_frame"],
            "recovery_success": rec["recovery_success"],
            "B1_error_cm": rec["B1_error_cm"],
            "B5_recovery_error_cm": rec["B5_recovery_error_cm"],
            "B5_operational_error_cm": rec["B5_operational_error_cm"],
            "Delta_B5_minus_B1_cm": rec["B5_minus_B1_cm"]
        })
        if np.isfinite(rec["B5_minus_B1_cm"]):
            recovery_differences.append(rec["B5_minus_B1_cm"])

    rec_mean_diff, rec_ci_low, rec_ci_high = compute_paired_bootstrap_ci(
        recovery_differences,
        n_bootstraps=args.bootstrap_samples,
        seed=args.seed
    )
    recovery_rows.append({
        "episode": "PAIRED_MEAN",
        "recovery_frame": np.nan,
        "recovery_success": np.nan,
        "B1_error_cm": np.nan,
        "B5_recovery_error_cm": np.nan,
        "B5_operational_error_cm": np.nan,
        "Delta_B5_minus_B1_cm": rec_mean_diff,
        "Paired_95CI_low_cm": rec_ci_low,
        "Paired_95CI_high_cm": rec_ci_high
    })
    recovery_df = pd.DataFrame(recovery_rows)
    recovery_path = f"./checkpoint2_paired_recovery_B5_vs_B1_threshold{args.risk_threshold}.csv"
    recovery_df.to_csv(recovery_path, index=False)
    print(f"Paired recovery-frame validation 已保存: {recovery_path}")
    print(f"   paired mean Δerror(B5-B1) = {rec_mean_diff:.4f} cm")
    print(f"   paired episode-bootstrap 95% CI = [{rec_ci_low:.4f}, {rec_ci_high:.4f}]")

    # ====================  5 大概率标定指标 ====================
    if last_prob_metrics_obs is not None and last_prob_metrics_prior is not None:
        prob_df = pd.DataFrame([
            {"Metric": "1. AUROC_obs (Risk Discrimination)",       "Value": f"{last_prob_metrics_obs['auroc_obs']:.4f}"},
            {"Metric": "2. AUPRC_obs (Precision-Recall AUC)",     "Value": f"{last_prob_metrics_obs['auprc_obs']:.4f}"},
            {"Metric": "3. Brier Score_obs (Probability MSE)",    "Value": f"{last_prob_metrics_obs['brier_obs']:.4f}"},
            {"Metric": "4. ECE_obs (Expected Calibration Error)", "Value": f"{last_prob_metrics_obs['ece_obs']:.4f}"},
            {"Metric": "5. Temp_Factor_obs (Temperature Scalar)", "Value": f"{last_prob_metrics_obs['temp_factor_obs']:.4f}"},
            {"Metric": "1. AUROC_prior (Risk Discrimination)",       "Value": f"{last_prob_metrics_prior['auroc_prior']:.4f}"},
            {"Metric": "2. AUPRC_prior (Precision-Recall AUC)",     "Value": f"{last_prob_metrics_prior['auprc_prior']:.4f}"},
            {"Metric": "3. Brier Score_prior (Probability MSE)",    "Value": f"{last_prob_metrics_prior['brier_prior']:.4f}"},
            {"Metric": "4. ECE_prior (Expected Calibration Error)", "Value": f"{last_prob_metrics_prior['ece_prior']:.4f}"},
            {"Metric": "5. Temp_Factor_prior (Temperature Scalar)", "Value": f"{last_prob_metrics_prior['temp_factor_prior']:.4f}"}
        ])

        prob_csv_path = f"./checkpoint2_probability_calibration_metrics_threshold{args.risk_threshold}.csv"
        prob_df.to_csv(prob_csv_path, index=False)
        print(f"5 大概率标定指标已成功导出为 CSV: {prob_csv_path}")

    # 5. 导出为全指标大 CSV 文件
    basename2 = os.path.basename(result_dirs[0])
    csv_out_path = f"./checkpoint2_full_metrics_episode_summary_{basename2}_threshold{args.risk_threshold}.csv"
    df_summary.to_csv(csv_out_path, index=False)
    print(f"全指标 Episode 级汇总表格已保存为: {csv_out_path}")
