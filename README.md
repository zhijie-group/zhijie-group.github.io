# [zhijie-group.github.io](https://zhijie-group.github.io/)

This git repo hosts all content displayed on the website of Deng AI Lab @ SJTU.


## How to build

1. Install Hugo > 0.12
2. Run the following commands:
```bash
git clone https://github.com/zhijie-group/zhijie-group.github.io.git
cd zhijie-group.github.io
hugo server
```


## How to generate publication list

1. Copy the latest publication json to the root folder.
2. Run the following
```bash
python gen_publications.py
```
3. Copy the output of this script to `content/publications.md`
