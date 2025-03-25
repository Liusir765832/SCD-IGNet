function render_rain_and_fog_effects()
    % 参数设置
    streak_root = 'data/Streaks_Garg06/'; % 雨滴条纹图片的根目录
    image_root = 'testfog/';
    img_file_list = dir([image_root, '*.png']); % 获取所有jpg图片文件
    num_of_file = length(img_file_list); % 图片文件数量
    
    % 输出目录
    output_root = 'out/';
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end
    
    % 遍历每张图片
    for fileindex = 1:num_of_file
        % 读取图片
        im = imread([image_root, img_file_list(fileindex).name]);
        [~, filename, ~] = fileparts(img_file_list(fileindex).name); % 获取图片文件名
        
        % 添加雾效果
        fog_intensity = 0.3; % 雾的浓度
        foggy_image = add_fog_effect(im, fog_intensity);
        
        % ...（这里插入雨滴效果渲染的代码）
        
        % 保存图片
        imwrite(uint8(foggy_image), [output_root, filename, '_foggy.png']);
    end
end

function foggy_image = add_fog_effect(image, fog_intensity)
    % image: 原始图像
    % fog_intensity: 雾的浓度，范围从0到1
    
    % 创建一个较小的灰度图像表示雾
    fog_image = randi([100, 150], [floor(size(image, 1)/10), floor(size(image, 2)/10)]);
    fog_image = imgaussfilt(fog_image, 5); % 使用高斯模糊使雾更加自然
    fog_image = imresize(fog_image, [size(image, 1), size(image, 2)]); % 调整雾图像大小以匹配原始图像
    
    % 将雾效果叠加到原始图像上
    foggy_image = double(image);
    for c = 1:3
        foggy_image(:,:,c) = foggy_image(:,:,c) + fog_image .* fog_intensity .* (1 - foggy_image(:,:,c)/255);
    end
    foggy_image = min(max(foggy_image, 0), 255); % 确保像素值在0-255范围内
end
