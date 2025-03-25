if exist('data/data/nightcity-fine/val/img/', 'dir')
    img_file_list = dir([image_root, '*.png']);
    if isempty(img_file_list)
        disp('指定的路径没有找到任何 .png 文件。');
    end
else
    disp('指定的路径不存在。');
end
