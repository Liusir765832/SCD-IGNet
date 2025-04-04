streak_root = 'data/Streaks_Garg06/'; 
image_root = 'test/';
img_file_list = dir([image_root, '*.png']);
num_of_strtype = 5; 
num_of_file = length(img_file_list); 
filelist_name = 'filelist'; 

if ~exist('newout', 'file')
    mkdir('newout')
end
if ~exist(filelist_name, 'file')
    mkdir(filelist_name)
end

rain_list_stats = fopen(sprintf('%s/rain.txt', filelist_name), 'w');
sparse_list_stats = fopen(sprintf('%s/sparse.txt', filelist_name), 'w'); 
middle_list_stats = fopen(sprintf('%s/mid.txt', filelist_name), 'w'); 
dense_list_stats = fopen(sprintf('%s/dense.txt', filelist_name), 'w');
clean_list_stats = fopen(sprintf('%s/norain.txt', filelist_name), 'w');

for fileindex = 1:num_of_file 
    im = imread([image_root, img_file_list(fileindex).name]); 
    [~, filename, ~] = fileparts(img_file_list(fileindex).name); 
    bh = size(im, 1); 
    bw = size(im, 2); 

    for str_index = num_of_strtype
        im = imread([image_root, img_file_list(fileindex).name]); 
        [~, filename, ~] = fileparts(img_file_list(fileindex).name); 
        bh = size(im, 1);
        bw = size(im, 2); 

        fog_image = randi([100, 150], [floor(size(im, 1)/10), floor(size(im, 2)/10)]);
        fog_image = imgaussfilt(fog_image, 5);
        fog_image = imresize(fog_image, [size(im, 1), size(im, 2)]);
        fog_intensity = 0.3;
        foggy_image = double(im);
        for c = 1:3
            foggy_image(:,:,c) = foggy_image(:,:,c) + fog_image .* fog_intensity .* (1 - foggy_image(:,:,c)/255);
        end
        foggy_image = min(max(foggy_image, 0), 255);
        clean_final = foggy_image;
        %clean_final = double(im); 
        st_final = zeros(bh, bw, 3); 
        str_file_list = dir([streak_root, '*.png']);

        stage_st_final = zeros(bh,bw,3);
        for i = 1:8
            strnum = randi(length(str_file_list));
            st = imread([streak_root, str_file_list(strnum).name]);
            st = st(4:end, :,:);
            st = imresize(st, 4); 
            newst = zeros(size(st));
            bwst = imbinarize(rgb2gray(st));
            mask = bwareafilt(bwst, [0, 1000]);
            
            for c = 1:3
                temp = st(:,:,c);
                temp = temp.*uint8(mask);
                newst(:,:,c) = imgaussfilt(temp, 1);
            end
            
            newst = imresize(newst, [bh, bw]); 
            tr = rand() * 0.2 + 0.25;
            clean_final = clean_final + double(newst) * tr;
            st_final = st_final + double(newst)*tr;
            stage_st_final = stage_st_final + double(newst)*tr;
        end
        
        imwrite(uint8(stage_st_final), sprintf('newout/str%s-type%d-dense.png', filename, str_index));
        fprintf(dense_list_stats, sprintf('newout/str%s-type%d-dense.png\n', filename, str_index));
        
        disp('dense'); disp(mean(stage_st_final(:)))

        stage_st_final = zeros(bh,bw,3);
        for i = 1:4
            strnum = randi(length(str_file_list));
            st = imread([streak_root, str_file_list(strnum).name]);
            st = st(4:end, :,:);
            st = imresize(st, 4);  
            newst = zeros(size(st));
            bwst = imbinarize(rgb2gray(st));
            mask = bwareafilt(bwst, [800, 4000]); 
            
            for c = 1:3
                temp = st(:,:,c);
                temp = temp.*uint8(mask);
                newst(:,:,c) = imgaussfilt(temp, 2);
            end
            
            newst = imresize(newst, 1); 
            sh = size(newst, 1); 
            sw = size(newst, 2);
            

            for iter = 1:6
                row = randi(sh - bh);
                col = randi(sw - bw);

                selected = newst(row:row+bh-1, col:col+bw-1, :);
                tr = rand() * 0.15 + 0.20;
                clean_final = clean_final + double(selected) * tr;
                st_final = st_final + double(selected)*tr; 
                stage_st_final = stage_st_final + double(selected)*tr;
            end
            
        end

        imwrite(uint8(stage_st_final),sprintf('newout/str%s-type%d-mid.png', filename, str_index)); 
        fprintf(middle_list_stats, sprintf('newout/str%s-type%d-mid.png\n', filename, str_index));
        

        disp('middle'); disp(mean(stage_st_final(:)))
        

        stage_st_final = zeros(bh,bw,3);
        for i = 1:3
            strnum = randi(length(str_file_list));
            st = imread([streak_root, str_file_list(strnum).name]);
            st = st(4:end, :,:);
            st = imresize(st, 4); 
            newst = zeros(size(st));
            bwst = imbinarize(rgb2gray(st));
            mask = bwareafilt(bwst, [2000, 6000]); 
            
            for c = 1:3
                temp = st(:,:,c);
                temp = temp.*uint8(mask);
                newst(:,:,c) = imgaussfilt(temp, 1);
            end
            
            sh = size(newst, 1); 
            sw = size(newst, 2);
            for iter = 1:6
                row = randi(sh - bh);
                col = randi(sw - bw);

                selected = newst(row:row+bh-1, col:col+bw-1, :);
                tr = rand() * 0.1 + 0.10;
                clean_final = clean_final + double(selected) * tr;
                st_final = st_final + double(selected)*tr; 
                stage_st_final = stage_st_final + double(selected)*tr;
            end

        end
        %imwrite(uint8(clean_final), sprintf('newout/%s.png',filename));
        %imwrite(uint8(stage_st_final), sprintf('newout/str%s-type%d-sparse.png', filename, str_index));
        %imwrite(uint8(st_final), sprintf('newout/str%s-type%d-all.png', filename, str_index));
        %fprintf(rain_list_stats, sprintf('newout/img%s-type%d-sparse.png\n',filename, str_index));
        %fprintf(sparse_list_stats, sprintf('newout/str%s-type%d-sparse.png\n', filename, str_index));
        %fprintf(clean_list_stats, sprintf('BSD300/%s\n', img_file_list(fileindex).name));
        % write rain image
        imwrite(uint8(clean_final), sprintf('newout/img%s-type%d-sparse.png',filename, str_index));
        % write sparse rain streak image
        imwrite(uint8(stage_st_final), sprintf('newout/str%s-type%d-sparse.png', filename, str_index));
        % write all rain streak image
        imwrite(uint8(st_final), sprintf('newout/str%s-type%d-all.png', filename, str_index));
        % write rain image into file list
        fprintf(rain_list_stats, sprintf('newout/img%s-type%d-sparse.png\n',filename, str_index));
        % write sparse
        fprintf(sparse_list_stats, sprintf('newout/str%s-type%d-sparse.png\n', filename, str_index));
        % write no rain
        fprintf(clean_list_stats, sprintf('BSD300/%s\n', img_file_list(fileindex).name));
        disp('sparse'); disp(mean(stage_st_final(:)));
        

        disp('sparse'); disp(mean(stage_st_final(:)));
    end
end

% 关