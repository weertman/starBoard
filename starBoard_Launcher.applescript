tell application "Terminal"
	activate
	do script "source ~/.zshrc 2>/dev/null; source /opt/anaconda3/etc/profile.d/conda.sh && conda activate starboard && cd /Users/starlab/Desktop/starBoard-main && python main.py"
end tell
