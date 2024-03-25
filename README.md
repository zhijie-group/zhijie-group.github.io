# [hao-ai-lab.github.io](https://hao-ai-lab.github.io/)

This git repo hosts all content displayed on the website of Hao AI Lab @ UCSD.


## How to build

1. Install Hugo > 0.12
2. Run the following commands:
```bash
git clone https://github.com/hao-ai-lab/hao-ai-lab.github.io.git
cd hao-ai-lab.github.io
hugo server
```


## How to generate publication list

1. Copy the latest publication json to the root folder.
2. Run the following
```bash
python gen_publications.py
```
3. Copy the output of this script to `content/publications.md`