# Part2  Image Classification - Cifar10

# Usage 
1. Create a new environment `conda create -n test python==3.8`, and activate the environment `conda activate test`
2. Install the requirements: `pip3 install -r requirements.txt`
3. Run the command to train the model: `python3 main.py`. 
If you need to change the data root, please specify in the parser. By default the root is `python3 main.py --data_root=./p2_data/annotations/train_annos.json`
    - 3 directories will be created: 
        1. acc_log/result
        2. curve/result
        3. save_dir/result 
    - We use the **best_model.pt** in `save_dir/result/best_model.pt` for evaluation
4. Run the command to evaluate the model: `python3 eval.py --path=save_dir/result/best_model.pt --test_anno=./p2_data/annotations/public_test_annos.json`
