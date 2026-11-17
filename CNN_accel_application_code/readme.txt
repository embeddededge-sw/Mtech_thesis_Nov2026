COMMON_APP - Unified CNN Inference Application

Combines HLS and Non-HLS CNN inference into a single application
with command-line mode selection. Both modes run all NUM_TEST_IMAGES (10000).

## Build

    make

Or manually:

    gcc -O2 -Wall -o cnn_common_test cnn_common_test.c test_images.c weights_array.c -I. -lm

## Usage

    sudo ./cnn_common_test --mode hls                  # HLS, all 10000 images
    sudo ./cnn_common_test --mode nonhls               # Non-HLS, all 10000 images
    sudo ./cnn_common_test --mode hls --images 100     # HLS, first 100 images
    sudo ./cnn_common_test --mode nonhls --images 500  # Non-HLS, first 500 images

    --mode   : hls or nonhls (required)
    --images : number of test images, 1-10000 (optional, default: 10000)

## Differences between modes

| Feature      | --mode hls                    | --mode nonhls                 |
|--------------|-------------------------------|-------------------------------|
| IOCTL        | CNN_IOCTL_RUN_INFERENCE       | CNN_IOCTL_RUN_NONHLS          |
| Weights      | Loaded via LOAD_WEIGHTS       | Skipped                       |
| Image prep   | Direct memcpy (int32)         | Masked to 8-bit (& 0xff)      |
| Test images  | NUM_TEST_IMAGES (10000)       | NUM_TEST_IMAGES (10000)       |
