# ML experiment for PyTorch, TensorFlow, and JAX on Apple Silicon

Google 官網文件範例是使用 Colab (T4 GPU) 環境，但總是要理解怎麼用原生的 GPU 做 Inference 才能往效能調校之路邁進。

這篇分享如何在 Apple Silicon 上面用 JAX/Flax inference Gemma2。

Google's official documentation provides examples using the Colab (T4 GPU) environment. However, to truly understand performance tuning, it's essential to grasp how to perform inference using native GPUs.

This article shares how to perform inference with Gemma2 using JAX/Flax on Apple Silicon.



## Part 1: Inference with Gemma using JAX and Flax on Apple Silicon

### Requirements
- Upgrade your system to macOS Sequoia 15.x and Xcode command line tools
- Prepare Kaggle API token

  - Save onto .env file

    ```
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_key
    ```

- Python and Pyenv

### Steps
1. Clone the repository:

    ```bash
    $ git clone https://github.com/jimmyliao/lab-mlx labmlx/
    $ cd lab_mlx
    ```

2. Create a virtual environment and update pip:

    ```bash
    $ pyenv virtualenv 3.11.6 lab_mlx
    $ pyenv activate lab_mlx
    ```

3. Switch to another folder and clone the repository of axlearn

    ```bash
    $ cd ../
    $ git clone https://github.com/apple/axlearn
    $ cd axlearn
    $ git reset --hard e4ff72cb377ec1f6e74484fe4525c2f8c205ad41
    $ pip install -e .
    $ cd ../lab-mlx
    ```

4. Install gemma library from google-deepmind

    ```bash
    $ pip install -q git+https://github.com/google-deepmind/gemma.git
    ```

5. Install the required packages for this lab:

    ```bash
    $ pip install -r requirements.txt
    ```

6. Validate the installation part 1:

    ```bash
    $ ENABLE_PJRT_COMPATIBILITY=1 python -c 'import jax; print(jax.numpy.arange(10))'

    Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    W0000 00:00:1735225492.671923  289381 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
    Metal device set to: Apple M2

    systemMemory: 24.00 GB
    maxCacheSize: 8.00 GB

    I0000 00:00:1735225492.687157  289381 service.cc:145] XLA service 0x6000031ec300 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
    I0000 00:00:1735225492.687186  289381 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
    I0000 00:00:1735225492.689097  289381 mps_client.cc:406] Using Simple allocator.
    I0000 00:00:1735225492.689129  289381 mps_client.cc:384] XLA backend will use up to 17179492352 bytes on device 0 for SimpleAllocator.
    [0 1 2 3 4 5 6 7 8 9]

    ```

7. Validate the installation part 2:

    ```bash
    $ ENABLE_PJRT_COMPATIBILITY=1 python -c 'import jax; jax.print_environment_info()'

    Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    W0000 00:00:1735225657.487601  297393 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
    Metal device set to: Apple M2

    systemMemory: 24.00 GB
    maxCacheSize: 8.00 GB

    I0000 00:00:1735225657.503874  297393 service.cc:145] XLA service 0x600003bdc800 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
    I0000 00:00:1735225657.504022  297393 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
    I0000 00:00:1735225657.506120  297393 mps_client.cc:406] Using Simple allocator.
    I0000 00:00:1735225657.506147  297393 mps_client.cc:384] XLA backend will use up to 17179492352 bytes on device 0 for SimpleAllocator.
    jax:    0.4.27
    jaxlib: 0.4.27
    numpy:  1.26.4
    python: 3.11.6 (main, Jan  4 2024, 13:46:14) [Clang 15.0.0 (clang-1500.1.0.2.5)]
    jax.devices (1 total, 1 local): [METAL(id=0)]
    process_count: 1
    platform: uname_result(system='Darwin', node='<YOUR_MAC_NAME>.local', release='24.2.0', version='Darwin Kernel Version 24.2.0: Fri Dec  6 18:51:28 PST 2024; root:xnu-11215.61.5~2/RELEASE_ARM64_T8112', machine='arm64')

    I0000 00:00:1735225657.558261  297393 mps_client.h:209] MetalClient destroyed.

    ```

8. Run the test script
    ```bash
    $ ENABLE_PJRT_COMPATIBILITY=1 python test.py

    GEMMA_PATH: /Users/<YOUR_NAME>/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b-it/1
    CKPT_PATH: /Users/<YOUR_NAME>/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b-it/1/gemma2-2b-it
    TOKENIZER_PATH: /Users/<YOUR_NAME>/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b-it/1/tokenizer.model
    WARNING:jax._src.xla_bridge:Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    W0000 00:00:1735225763.060748  301175 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
    Metal device set to: Apple M2

    systemMemory: 24.00 GB
    maxCacheSize: 8.00 GB

    I0000 00:00:1735225763.082880  301175 service.cc:145] XLA service 0x60000174c400 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
    I0000 00:00:1735225763.083035  301175 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
    I0000 00:00:1735225763.085140  301175 mps_client.cc:406] Using Simple allocator.
    I0000 00:00:1735225763.085169  301175 mps_client.cc:384] XLA backend will use up to 17179492352 bytes on device 0 for SimpleAllocator.
    Vocab size: 256000
    Transformer ready!
    Sampler ready!
    Prompting... ['台灣的總人口數是多少？並列出統計年份。']
    Reply: ['台灣的總人口數是多少？並列出統計年份。\n\n**答案：**\n\n截至2023年，台灣總人口數約為2380萬。\n\n以下是台灣人口統計年份：']
    Prompt:
    台灣的總人口數是多少？並列出統計年份。
    Output:
    台灣的總人口數是多少？並列出統計年份。

    **答案：**

    截至2023年，台灣總人口數約為2380萬。

    以下是台灣人口統計年份：

    ```


### Note / References
- Google Gemma Jax Inference site: [link](https://ai.google.dev/gemma/docs/jax_inference)
- Apple JAX Metal site: [link](https://developer.apple.com/metal/jax/)
- According to [this](https://github.com/jax-ml/jax/issues/21383#issuecomment-2130232491), you need to set up the environment variable `ENABLE_PJRT_COMPATIBILITY=1` to run the JAX code on Apple Silicon if you encounter the error message 
    ```bash
    RuntimeError: Unable to initialize backend 'METAL': INVALID_ARGUMENT: Mismatched PJRT plugin PJRT API version (0.47) and framework PJRT API version 0.51). (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)
    ```

- cleanup all installed packages
    ```bash
    pip uninstall -r requirements.txt --yes
    ```
- Next MLX Exploration reference: [https://github.com/ml-explore/mlx/](https://github.com/ml-explore/mlx/)
