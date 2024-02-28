import torch
import os
import ssl

def download_process_data(path="colab_demo"):
    os.makedirs(path, exist_ok=True)
    ssl._create_default_https_context = ssl._create_unverified_context
    print("Downloading data")
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom1.pth', os.path.join(path, 'lsun_bedroom1.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom2.pth', os.path.join(path, 'lsun_bedroom2.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom3.pth', os.path.join(path, 'lsun_bedroom3.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_edit.pth', os.path.join(path, 'lsun_edit.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_church.pth', os.path.join(path, 'lsun_church.pth'))
    print("Data downloaded")
