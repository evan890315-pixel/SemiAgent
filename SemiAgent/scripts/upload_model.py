# 另存為 scripts/upload_models.py 然後執行
from huggingface_hub import HfApi

api = HfApi()

# 上傳分類器
api.upload_folder(
    folder_path=r"C:\\Users\\evan8\\OneDrive\\桌面\SemiAgent_v3\SemiAgent\\models",
    repo_id="Evanjia1001/semiagent",
    repo_type="model",
)
print("✅ 分類器上傳完成")

# 上傳生成器
api.upload_folder(
    folder_path=r"C:\\Users\\evan8\\OneDrive\\桌面\SemiAgent_v3\SemiAgent\\models",
    repo_id="Evanjia1001/semiagent",
    repo_type="model",
)
print("✅ 生成器上傳完成")