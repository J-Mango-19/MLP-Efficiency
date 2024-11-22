# download 60000 mnist dataset images and store in csv file under data directory

URL="https://pjreddie.com/media/files/mnist_train.csv"
DATA_DIR="./data"
FILENAME="MNIST_data.csv"

if [  -e "$DATA_DIR/$FILENAME" ]; then
    echo "MNIST dataset found at $DATA_DIR/$FILENAME"
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "Downloading MNIST dataset..."
if command -v wget > /dev/null; then
    wget -O "$DATA_DIR/$FILENAME" "$URL"
elif command -v curl > /dev/null; then
    curl -o "$DATA_DIR/$FILENAME" "$URL"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them and try again."
    exit 1
fi

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully. The dataset is stored in $DATA_DIR/$FILENAME"
else
    echo "Error: MNIST dataset Download failed."
    exit 1
fi


