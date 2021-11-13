find . -name '*.c' -o -name '*.java' -o -name '*.h' -o -name '*.cpp' -o -name '*.cc' -o -name '.sh' -o -name '.mk' -o -name '*.yml' > /home/chenwei/Documents/SelfStudy/Code/Code_Of_Android_ECG_SEG/ctagscope/ecg_seg.files 
cscope -Rbkq -i ~/Documents/SelfStudy/Code/Code_Of_Android_ECG_SEG/ctagscope/ecg_seg.files -f ~/Documents/SelfStudy/Code/Code_Of_Android_ECG_SEG/ctagscope/ecg_seg.cout
ctags -R -f /home/chenwei/Documents/SelfStudy/Code/Code_Of_Android_ECG_SEG/ctagscope/ecg_seg.tag --tag-relative=yes

