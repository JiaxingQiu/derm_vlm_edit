## DermEdit: Editing VLMs using Real Dermotologist Corrections

### Requirements

```bash
cd derm_vlm_edit
conda create -n dermedit python=3.10 -y
conda activate dermedit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec uninstall dermedit -y
python -m ipykernel install --user --name dermedit --display-name "dermedit"
jupyter kernelspec list
```
