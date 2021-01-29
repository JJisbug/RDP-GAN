clear;

% orig_data = importdata('adult.data');
% a=csvread('Adultall.csv');
% data = fopen('Adultall.csv', 'w');

% income_data = zeros(1, length(orig_data));
% for i = 1:length(orig_data)
%     meta_data = orig_data(i);
%     if contains(meta_data,'>')
%         data.data(i,11) = 1;
%     end
% end
%load('raw_data.mat');
%load('agg_data.mat');
%data=Raw_data;
%data=Agg_data;
data = importdata('C:\Users\hp\Desktop\dp_gan\tableGAN-master\samples\Adult\Adult_None_std0.0_epoch122_fake.csv');

%[m, n] = size(data.data);
[m, n] = size(data);
encode_data = zeros(m, 1);
i_index = 0;
for i = 1:n
    if i < n
        %max_value = max(data.data(:,i));
        max_value = max(data(:,i));
        num_bit = ceil(log2(max_value));
        for j = 1:m
            %meta_data = data.data(j,i)-1;
            meta_data = data(j,i)-1;
            for k = num_bit:-1:1
                if meta_data > 2^(k-1)-1
                    encode_data(j,num_bit-k+1+i_index) = 1;
                    meta_data = meta_data - 2^(k-1);
                end
            end
        end
        i_index = i_index+num_bit;
    else
        for j = 1:m
            %encode_data(j,1+i_index) = data.data(j,i);
            encode_data(j,1+i_index) = data(j,i);
        end
    end
end

tab_data=table(encode_data);
writetable(tab_data,'encode_Raw_data.csv');
%writetable(tab_data,'encode_Agg_data.csv');
%save('./adult.csv', data)
