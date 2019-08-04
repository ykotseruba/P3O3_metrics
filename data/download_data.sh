echo "Downloading P3 stimuli..."
wget http://data.nvision2.eecs.yorku.ca/P3O3/data/P3/P3_data.zip 
mkdir P3
unzip P3_data.zip -d P3
rm P3_data.zip

echo "Downloading P3 train/val/test splits..."
wget http://data.nvision2.eecs.yorku.ca/P3O3/data/P3/P3_train_test_split.zip
unzip P3_train_test_split.zip -d P3
rm P3_train_test_split.zip

echo "Downloading O3 stimuli..."
wget http://data.nvision2.eecs.yorku.ca/data/O3/O3_data.zip
mkdir O3
unzip O3_data.zip -d O3
rm O3_data.zip

echo "Downloading O3 train/val/test splits..."
wget http://data.nvision2.eecs.yorku.ca/P3O3/data/O3/O3_train_test_split.zip
unzip O3_train_test_split.zip -d O3
rm O3_train_test_split.zip
