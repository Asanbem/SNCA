# SNCA

This is the code repository of the ariticle **Boosting adversarial transferability via spatial-neighborhood composite**

# Environment Setup

Since our mechanisms are built upon established frameworks, please follow the original setup instructions:

Refer to the official [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) repository for dependency and dataset setup.

# Usage

Assuming you are in the `TransferAttack` root directory with properly configured datasets, run:

```
# Generate adversarial examples using SNCA
python main.py --input_dir ./data --output_dir adv_data/snca/inception_v3 --attack snca --model inception_v3

# Evaluate the generated adversarial examples
python main.py --input_dir ./data --output_dir adv_data/snca/inception_v3 --eval
```

# Acknowledgements

We would like to thank the authors of the following outstanding open-source repositories, which significantly facilitated our research and codebase construction: [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)