import os
from pathlib import Path
from typing import List
from random import randint

class MyList:
    def __init__(self, path: Path, lst: List[str]) -> None:
        self.path = path
        self.lst = lst
        self._index = None

    def select(self, index):
        """index <item_index> | word <keyword>"""
        if callable(index):
            ids = [i for i,s in enumerate(self.lst) if index(s)]
            if len(ids) == 1:
                ids = ids[0]
            self._index = ids
        else:
            self._index = index
        return self
    
    def assert_one_index(self):
        idx = self._index
        if idx is None:
            raise ValueError("缺少 index")
        elif not isinstance(idx, int):
            raise ValueError(f"index数量太多: {idx}")    
        return self

    def add_before(self, item):
        idx = self._index
        if idx is None:
            idx = len(self.lst)
        self.lst[idx:idx] = [item]
        self._index = None
        return self

    def delete(self):
        idx = self._index
        del self.lst[idx]
        self._index = None
        return self

    def random(self):
        if len(self.lst) == 0:
            raise ValueError(f"{self.path} 还空空如也，无法取出一个。")
        self._index = randint(0, len(self.lst) - 1)
        return self

    def log(self, message_func):
        print(message_func(self))
        return self

    def print(self, number=True, chain=False):
        # logger.debug(f"printing {self.path, self.lst, self._index ,self.index_list(), self.as_list()}")
        result = "\n".join(
            (f"{i+1}:" if number else "") + f"{x}" for i,x in enumerate(self.lst) if i in self.index_list()
        )
        if chain:
            print(result)
            return self
        else:
            return result
    
    def index_list(self):
        idx = self._index
        idx = [idx] if isinstance(idx, int) else idx
        idx = range(0, len(self.lst)) if idx is None else idx
        return idx

    def as_list(self):
        return [x for i,x in enumerate(self.lst) if i in self.index_list()]

    def shrink(self):
        return MyList(self.path, self.as_list())

    def strip_path(self):
        self.path = Path(".") / self.path.parts[-1]
        return self

    def __iter__(self):
        return iter(self.as_list())

    def merge(self, other):
        self.lst.extend(other)
        return self

    def dedup(self):
        rev_map = {k:i for i,k in reversed(list(enumerate(self)))}
        all_index = set(range(0,len(self.as_list())))
        deduped = all_index - set(rev_map.values())
        deduped = list(deduped)
        deduped.sort()
        print(f"de-duplicated '{self.path}': ")
        print(self.select(deduped).print())
        return self.select(all_index - set(deduped))

    @classmethod
    def load_file(cls, file_path):
        file_path = Path(file_path)
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                return MyList(file_path, [l.strip() for l in f if l.strip() != ""])
        else:
            return MyList(file_path, [])

    @classmethod
    def load_dir_name(cls, file_path):
        file_path = Path(file_path)
        if file_path.is_dir():
            return MyList(file_path,[str(dir_file) for dir_file in file_path.iterdir() if dir_file.is_file()])
        else:
            return MyList(file_path, [])

    @classmethod
    def load_dir(cls, file_path):
        file_path = Path(file_path)
        if file_path.is_dir():
            return [MyList.load_file(dir_file) for dir_file in file_path.iterdir() if dir_file.is_file()]
        else:
            return []
    
    def save(self):
        # logger.debug(f"saving {self.path, self.lst, self._index ,self.index_list(), self.as_list()}")
        with open(self.path, "w", encoding="utf-8") as f:
            for l in self:
                print(l.strip(), file=f)
    
    def remove_self(self):
        if self.path.is_file():
            print(f"remove {len(self.as_list())} items from '{self.path}'")
            os.remove(self.path)
        else:
            print(f"要移除的 '{self.path}' 不存在。")