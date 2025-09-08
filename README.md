# AI 市場情報分析助理

整合 **Yahoo Finance** 股價 / 新聞 + **技術指標**（SMA、RSI、KD、MACD…）  
並使用 **Gemini** 自動標註新聞情緒與撰寫分析總結（僅供教育用途，不構成投資建議）。

> 線上部署可用 **Streamlit Community Cloud**，或本機執行。

---

## 目錄結構

├─ stock_analyzer.py # 主程式（Streamlit app）
├─ requirements.txt # 套件需求
├─ news_classify_prompt.txt # 新聞情緒標註 Prompt（可改）
├─ final_report_prompt.txt # 最終報告 Prompt（可改）
├─ README.md # 本說明
├─ .gitignore # Git 忽略規則（請確認 .env 已被忽略）
└─ .env # 本機環境變數（不要上傳）