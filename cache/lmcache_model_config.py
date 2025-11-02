"""
LMCache模型配置文件
用于根据实际使用的模型设置正确的LMCache参数
"""

import torch
from typing import Dict, Any, Optional


class ModelConfig:
    """模型配置基类"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def get_lmcache_metadata(self) -> Dict[str, Any]:
        """获取LMCache元数据"""
        raise NotImplementedError
    
    def get_gpu_connector_params(self) -> Dict[str, Any]:
        """获取GPU连接器参数"""
        raise NotImplementedError
    
    def get_auto_hyperparams(self, gpu_memory_gb: float = 24.0) -> Dict[str, Any]:
        """
        根据模型特性和GPU内存自动设置超参数
        
        Args:
            gpu_memory_gb: GPU内存大小（GB）
            
        Returns:
            自动配置的超参数字典
        """
        raise NotImplementedError
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        raise NotImplementedError


class Qwen25VL7BConfig(ModelConfig):
    """Qwen2.5-VL-7B模型配置"""
    
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-VL-7B-Instruct")
    
    def get_lmcache_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "qwen2.5-vl-7b",
            "world_size": 1,
            "worker_id": 0,
            "fmt": "vllm",
            "kv_dtype": torch.bfloat16,
            "kv_shape": (32, 2, 256, 32, 128),  # (num_layers, 2, chunk_size, num_kv_head, head_size)
            "use_mla": False
        }
    
    def get_gpu_connector_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim_size": 4096,  # 32 * 128 = 4096
            "num_layers": 32,
            "chunk_size": 256,
            "kv_dtype": torch.bfloat16,
            "device": "cuda",
            "use_mla": False
        }
    
    def get_auto_hyperparams(self, gpu_memory_gb: float = 24.0) -> Dict[str, Any]:
        """
        为Qwen2.5-VL-7B模型自动设置超参数
        
        基于模型特性：
        - 7B参数模型，相对较小
        - 32层，每层4096维
        - 适合中等GPU内存
        """
        
        # 基础配置
        base_config = {
            'chunk_size': 256,
            'enable_blending': True,
            'blend_mode': 'graphrag',
            'blend_separator': '[LMCACHE_BLEND_SEP]',
            'blend_add_special_in_precomp': False,
            'enable_p2p': False,
            'pipelined_backend': False,
            'save_decode_cache': False,
            'remote_serde': 'torch'
        }
        
        # 根据GPU内存自动调整
        if gpu_memory_gb >= 40:
            # 大内存GPU：可以设置更大的缓存
            base_config.update({
                'max_local_cache_size': 20.0,
                'blend_recompute_ratio': 0.10,  # 更少的重新计算
                'blend_min_tokens': 128,         # 更短的序列也能混合
                'local_device': 'cuda'
            })
        elif gpu_memory_gb >= 24:
            # 中等内存GPU：平衡配置
            base_config.update({
                'max_local_cache_size': 12.0,
                'blend_recompute_ratio': 0.15,
                'blend_min_tokens': 256,
                'local_device': 'cuda'
            })
        elif gpu_memory_gb >= 16:
            # 小内存GPU：保守配置
            base_config.update({
                'max_local_cache_size': 8.0,
                'blend_recompute_ratio': 0.20,  # 更多重新计算以节省内存
                'blend_min_tokens': 512,        # 更长的序列才混合
                'local_device': 'cuda'
            })
        else:
            # 极小内存GPU：使用CPU缓存
            base_config.update({
                'max_local_cache_size': 5.0,
                'blend_recompute_ratio': 0.25,
                'blend_min_tokens': 1024,
                'local_device': 'cpu'
            })
        
        return base_config
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取Qwen2.5-VL-7B模型信息"""
        return {
            'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'model_size': '7B',
            'architecture': 'Qwen2.5-VL',
            'num_layers': 32,
            'num_kv_head': 32,
            'head_size': 128,
            'hidden_dim': 4096,
            'vocab_size': 151936,
            'max_seq_len': 32768,
            'multimodal': True,
            'vision_tower': 'Qwen2.5-VL',
            'recommended_gpu_memory': '16GB+',
            'optimal_chunk_size': 256,
            'optimal_blend_ratio': 0.15
        }


class Qwen25VL32BConfig(ModelConfig):
    """Qwen2.5-VL-32B模型配置"""
    
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-VL-32B-Instruct")
    
    def get_lmcache_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "qwen2.5-vl-32b",
            "world_size": 1,
            "worker_id": 0,
            "fmt": "vllm",
            "kv_dtype": torch.bfloat16,
            "kv_shape": (64, 2, 256, 32, 128),  # (num_layers, 2, chunk_size, num_kv_head, head_size)
            "use_mla": False
        }
    
    def get_gpu_connector_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim_size": 4096,  # 32 * 128 = 4096
            "num_layers": 64,
            "chunk_size": 256,
            "kv_dtype": torch.bfloat16,
            "device": "cuda",
            "use_mla": False
        }
    
    def get_auto_hyperparams(self, gpu_memory_gb: float = 24.0) -> Dict[str, Any]:
        """
        为Qwen2.5-VL-32B模型自动设置超参数
        
        基于模型特性：
        - 32B参数模型，较大
        - 64层，每层4096维
        - 需要大GPU内存
        """
        
        # 基础配置
        base_config = {
            'chunk_size': 256,
            'enable_blending': True,
            'blend_mode': 'graphrag',
            'blend_separator': '[LMCACHE_BLEND_SEP]',
            'blend_add_special_in_precomp': False,
            'enable_p2p': False,
            'pipelined_backend': False,
            'save_decode_cache': False,
            'remote_serde': 'torch'
        }
        
        # 根据GPU内存自动调整（32B模型需要更多内存）
        if gpu_memory_gb >= 80:
            # 超大内存GPU
            base_config.update({
                'max_local_cache_size': 30.0,
                'blend_recompute_ratio': 0.08,
                'blend_min_tokens': 128,
                'local_device': 'cuda'
            })
        elif gpu_memory_gb >= 48:
            # 大内存GPU
            base_config.update({
                'max_local_cache_size': 20.0,
                'blend_recompute_ratio': 0.12,
                'blend_min_tokens': 256,
                'local_device': 'cuda'
            })
        elif gpu_memory_gb >= 32:
            # 中等内存GPU：保守配置
            base_config.update({
                'max_local_cache_size': 15.0,
                'blend_recompute_ratio': 0.18,
                'blend_min_tokens': 512,
                'local_device': 'cuda'
            })
        else:
            # 小内存GPU：使用CPU缓存
            base_config.update({
                'max_local_cache_size': 8.0,
                'blend_recompute_ratio': 0.25,
                'blend_min_tokens': 1024,
                'local_device': 'cpu'
            })
        
        return base_config
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取Qwen2.5-VL-32B模型信息"""
        return {
            'model_name': 'Qwen/Qwen2.5-VL-32B-Instruct',
            'model_size': '32B',
            'architecture': 'Qwen2.5-VL',
            'num_layers': 64,
            'num_kv_head': 32,
            'head_size': 128,
            'hidden_dim': 4096,
            'vocab_size': 151936,
            'max_seq_len': 32768,
            'multimodal': True,
            'vision_tower': 'Qwen2.5-VL',
            'recommended_gpu_memory': '48GB+',
            'optimal_chunk_size': 256,
            'optimal_blend_ratio': 0.12
        }


class MiMoVL7BConfig(ModelConfig):
    """MiMo-VL-7B模型配置"""
    
    def __init__(self):
        super().__init__("XiaomiMiMo/MiMo-VL-7B-RL")
    
    def get_lmcache_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "mimo-vl-7b",
            "world_size": 1,
            "worker_id": 0,
            "fmt": "vllm",
            "kv_dtype": torch.float16,
            "kv_shape": (32, 2, 256, 32, 128),  # (num_layers, 2, chunk_size, num_kv_head, head_size)
            "use_mla": False
        }
    
    def get_gpu_connector_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim_size": 4096,  # 32 * 128 = 4096
            "num_layers": 32,
            "chunk_size": 256,
            "kv_dtype": torch.float16,
            "device": "cuda",
            "use_mla": False
        }


class LlavaNext8BConfig(ModelConfig):
    """LLaVA-NeXT-8B模型配置"""
    
    def __init__(self):
        super().__init__("llava-hf/llama3-llava-next-8b-hf")
    
    def get_lmcache_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "llava-next-8b",
            "world_size": 1,
            "worker_id": 0,
            "fmt": "vllm",
            "kv_dtype": torch.float16,
            "kv_shape": (32, 2, 256, 32, 128),  # (num_layers, 2, chunk_size, num_kv_head, head_size)
            "use_mla": False
        }
    
    def get_gpu_connector_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim_size": 4096,  # 32 * 128 = 4096
            "num_layers": 32,
            "chunk_size": 256,
            "kv_dtype": torch.float16,
            "device": "cuda",
            "use_mla": False
        }


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    根据模型名称获取对应的配置
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        对应的模型配置对象，如果找不到则返回None
    """
    
    # 模型名称映射
    model_configs = {
        "Qwen/Qwen2.5-VL-7B-Instruct": Qwen25VL7BConfig(),
        "Qwen/Qwen2.5-VL-32B-Instruct": Qwen25VL32BConfig(),
        "XiaomiMiMo/MiMo-VL-7B-RL": MiMoVL7BConfig(),
        "llava-hf/llama3-llava-next-8b-hf": LlavaNext8BConfig(),
        "qwen": Qwen25VL7BConfig(),  # 默认使用7B版本
        "mimo": MiMoVL7BConfig(),
        "llava": LlavaNext8BConfig(),
    }
    
    # 精确匹配
    if model_name in model_configs:
        return model_configs[model_name]
    
    # 模糊匹配
    for key, config in model_configs.items():
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            return config
    
    return None


def create_custom_model_config(
    model_name: str,
    num_layers: int,
    num_kv_head: int,
    head_size: int,
    kv_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 256
) -> ModelConfig:
    """
    创建自定义模型配置
    
    Args:
        model_name: 模型名称
        num_layers: 层数
        num_kv_head: KV头数
        head_size: 头大小
        kv_dtype: KV数据类型
        chunk_size: 块大小
        
    Returns:
        自定义模型配置对象
    """
    
    class CustomModelConfig(ModelConfig):
        def __init__(self, model_name, num_layers, num_kv_head, head_size, kv_dtype, chunk_size):
            super().__init__(model_name)
            self.num_layers = num_layers
            self.num_kv_head = num_kv_head
            self.head_size = head_size
            self.kv_dtype = kv_dtype
            self.chunk_size = chunk_size
        
        def get_lmcache_metadata(self) -> Dict[str, Any]:
            return {
                "model_name": model_name.lower().replace("/", "-").replace("_", "-"),
                "world_size": 1,
                "worker_id": 0,
                "fmt": "vllm",
                "kv_dtype": self.kv_dtype,
                "kv_shape": (self.num_layers, 2, self.chunk_size, self.num_kv_head, self.head_size),
                "use_mla": False
            }
        
        def get_gpu_connector_params(self) -> Dict[str, Any]:
            return {
                "hidden_dim_size": self.num_kv_head * self.head_size,
                "num_layers": self.num_layers,
                "chunk_size": self.chunk_size,
                "kv_dtype": self.kv_dtype,
                "device": "cuda",
                "use_mla": False
            }
    
    return CustomModelConfig(model_name, num_layers, num_kv_head, head_size, kv_dtype, chunk_size)


def print_model_config_info(model_config: ModelConfig):
    """
    打印模型配置信息
    
    Args:
        model_config: 模型配置对象
    """
    
    print(f"Model: {model_config.model_name}")
    print("LMCache Metadata:")
    metadata = model_config.get_lmcache_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print("GPU Connector Parameters:")
    gpu_params = model_config.get_gpu_connector_params()
    for key, value in gpu_params.items():
        print(f"  {key}: {value}")


# 使用示例
if __name__ == "__main__":
    # 获取预定义模型配置
    qwen_config = get_model_config("Qwen/Qwen2.5-VL-7B-Instruct")
    if qwen_config:
        print_model_config_info(qwen_config)
    
    # 创建自定义模型配置
    custom_config = create_custom_model_config(
        "custom-model",
        num_layers=48,
        num_kv_head=64,
        head_size=128,
        kv_dtype=torch.float16,
        chunk_size=512
    )
    print("\nCustom Model Config:")
    print_model_config_info(custom_config)


def detect_gpu_memory() -> float:
    """
    检测GPU内存大小
    
    Returns:
        GPU内存大小（GB），如果检测失败返回默认值24.0
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
            return gpu_memory_gb
        else:
            print("CUDA not available, using default GPU memory: 24.0 GB")
            return 24.0
    except Exception as e:
        print(f"Failed to detect GPU memory: {e}, using default: 24.0 GB")
        return 24.0


def get_optimal_config_for_model(
    model_name: str, 
    gpu_memory_gb: Optional[float] = None,
    auto_detect_gpu: bool = True
) -> Dict[str, Any]:
    """
    为指定模型获取最优配置
    
    Args:
        model_name: 模型名称
        gpu_memory_gb: GPU内存大小（GB），如果为None则自动检测
        auto_detect_gpu: 是否自动检测GPU内存
        
    Returns:
        包含所有配置的字典
    """
    
    # 获取模型配置
    model_config = get_model_config(model_name)
    if model_config is None:
        print(f"Warning: No predefined config found for model {model_name}")
        return {}
    
    # 检测GPU内存
    if gpu_memory_gb is None and auto_detect_gpu:
        gpu_memory_gb = detect_gpu_memory()
    elif gpu_memory_gb is None:
        gpu_memory_gb = 24.0  # 默认值
    
    # 获取自动超参数
    hyperparams = model_config.get_auto_hyperparams(gpu_memory_gb)
    
    # 获取GPU连接器参数
    gpu_params = model_config.get_gpu_connector_params()
    
    # 获取模型信息
    model_info = model_config.get_model_info()
    
    # 合并所有配置
    config = {
        'model_info': model_info,
        'gpu_connector_params': gpu_params,
        'hyperparams': hyperparams,
        'detected_gpu_memory_gb': gpu_memory_gb
    }
    
    return config


def print_optimal_config(config: Dict[str, Any]):
    """
    打印最优配置信息
    
    Args:
        config: 配置字典
    """
    
    if not config:
        print("No configuration available")
        return
    
    print("=" * 60)
    print("LMCache Optimal Configuration")
    print("=" * 60)
    
    # 模型信息
    if 'model_info' in config:
        print("\n📋 Model Information:")
        model_info = config['model_info']
        for key, value in model_info.items():
            print(f"  {key}: {value}")
    
    # GPU连接器参数
    if 'gpu_connector_params' in config:
        print("\n🔧 GPU Connector Parameters:")
        gpu_params = config['gpu_connector_params']
        for key, value in gpu_params.items():
            print(f"  {key}: {value}")
    
    # 超参数
    if 'hyperparams' in config:
        print("\n⚙️  Auto-configured Hyperparameters:")
        hyperparams = config['hyperparams']
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    
    # GPU内存信息
    if 'detected_gpu_memory_gb' in config:
        print(f"\n💾 Detected GPU Memory: {config['detected_gpu_memory_gb']:.1f} GB")
    
    print("=" * 60)


def create_lmcache_config_file(
    model_name: str,
    output_path: str = "lmcache_optimal_config.yaml",
    gpu_memory_gb: Optional[float] = None
):
    """
    创建LMCache最优配置文件
    
    Args:
        model_name: 模型名称
        output_path: 输出文件路径
        gpu_memory_gb: GPU内存大小（GB）
    """
    
    try:
        import yaml
        
        # 获取最优配置
        config = get_optimal_config_for_model(model_name, gpu_memory_gb)
        
        if not config:
            print("Failed to get configuration")
            return
        
        # 准备YAML配置
        yaml_config = {
            'model': config.get('model_info', {}),
            'gpu_connector': config.get('gpu_connector_params', {}),
            'hyperparameters': config.get('hyperparams', {}),
            'gpu_memory_gb': config.get('detected_gpu_memory_gb', 24.0)
        }
        
        # 写入YAML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        print(f"Configuration saved to: {output_path}")
        
    except ImportError:
        print("PyYAML not available, cannot create config file")
    except Exception as e:
        print(f"Failed to create config file: {e}")


# 便捷函数
def auto_configure_qwen_7b(gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """为Qwen2.5-VL-7B模型自动配置"""
    return get_optimal_config_for_model("Qwen/Qwen2.5-VL-7B-Instruct", gpu_memory_gb)


def auto_configure_qwen_32b(gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """为Qwen2.5-VL-32B模型自动配置"""
    return get_optimal_config_for_model("Qwen/Qwen2.5-VL-32B-Instruct", gpu_memory_gb)


def auto_configure_mimo_7b(gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """为MiMo-VL-7B模型自动配置"""
    return get_optimal_config_for_model("XiaomiMiMo/MiMo-VL-7B-RL", gpu_memory_gb)
