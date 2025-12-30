import json
from pathlib import Path
from datetime import datetime

class PipelineState:
    """流水线状态管理器，用于追踪模块执行状态和缓存"""

    def __init__(self, state_file=None):
        if state_file is None:
            from src.config import Config
            state_file = Config.CACHE_FOLDER / 'pipeline_state.json'

        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self):
        """加载状态文件"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'completed_modules': [],
            'module_metadata': {},
            'last_run': None
        }

    def _save_state(self):
        """保存状态到文件"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def is_completed(self, module_name):
        """检查模块是否已完成"""
        return module_name in self.state['completed_modules']

    def mark_completed(self, module_name, metadata=None):
        """标记模块为已完成"""
        if module_name not in self.state['completed_modules']:
            self.state['completed_modules'].append(module_name)

        self.state['module_metadata'][module_name] = {
            'completed_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.state['last_run'] = datetime.now().isoformat()
        self._save_state()

    def get_metadata(self, module_name):
        """获取模块的元数据"""
        return self.state['module_metadata'].get(module_name, {})

    def reset(self, module_name=None):
        """重置状态（可选择性重置特定模块）"""
        if module_name:
            if module_name in self.state['completed_modules']:
                self.state['completed_modules'].remove(module_name)
            if module_name in self.state['module_metadata']:
                del self.state['module_metadata'][module_name]
        else:
            self.state = {
                'completed_modules': [],
                'module_metadata': {},
                'last_run': None
            }
        self._save_state()

    def get_cache_path(self, cache_name):
        """获取缓存文件路径"""
        from src.config import Config
        return Config.CACHE_FOLDER / cache_name
