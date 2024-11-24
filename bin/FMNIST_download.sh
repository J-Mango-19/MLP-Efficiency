#!/bin/bash

# Downloads 60000 Fashion-MNIST images and stores them in data/MNIST_data.csv

DATA_DIR="./data"
FILENAME="MNIST_data.csv"

if [ -e "$DATA_DIR/$FILENAME" ]; then
    echo "Dataset found at $DATA_DIR/$FILENAME"
    exit 0
fi

mkdir -p "$DATA_DIR"

# Download the compressed files
echo "Downloading Fashion MNIST dataset..."

if command -v wget > /dev/null; then
    wget -P "$DATA_DIR" http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    wget -P "$DATA_DIR" http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

elif command -v curl > /dev/null; then
    curl -o "$DATA_DIR/$(basename http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)" http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    curl -o "$DATA_DIR/$(basename http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)" http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
fi

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Error: Fashion MNIST dataset download failed"
    exit 1
fi

# Unzip the files
cd "$DATA_DIR"
gunzip *.gz

echo "Converting training data to CSV..."
(
    paste -d, \
        <(hexdump -v -s 8 -e '1/1 "%u\n"' train-labels-idx1-ubyte | head -n 60000) \
        <(hexdump -v -s 16 -e '1/1 "%u\n"' train-images-idx3-ubyte | \
          awk 'BEGIN{ORS=""} NR%784==1{if(NR>1)print "\n"}{printf "%s%s", $0, (NR%784==0)?"":","}' | \
          head -n 60000)
) > MNIST_data.csv

# Clean up binary files
rm *-ubyte

