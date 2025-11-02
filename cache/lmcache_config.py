"""
LMCache配置文件
用于设置LMCache的环境变量和配置参数
"""

import os
from typing import Dict, Any


def setup_lmcache_environment(config: Dict[str, Any]):
    """
    设置LMCache的环境变量
    
    Args:
        config: 包含LMCache配置的字典
    """
    
    # LMCache环境变量映射
    env_mapping = {
        'chunk_size': 'LMCACHE_CHUNK_SIZE',
        'local_device': 'LMCACHE_LOCAL_DEVICE',
        'max_local_cache_size': 'LMCACHE_MAX_LOCAL_CACHE_SIZE',
        'enable_blending': 'LMCACHE_ENABLE_BLENDING',
        'blend_recompute_ratio': 'LMCACHE_BLEND_RECOMPUTE_RATIO',
        'blend_min_tokens': 'LMCACHE_BLEND_MIN_TOKENS',
        'blend_separator': 'LMCACHE_BLEND_SEPARATOR',
        'blend_add_special_in_precomp': 'LMCACHE_BLEND_ADD_SPECIAL_IN_PRECOMP',
        'enable_p2p': 'LMCACHE_ENABLE_P2P',
        'remote_url': 'LMCACHE_REMOTE_URL',
        'remote_serde': 'LMCACHE_REMOTE_SERDE',
        'pipelined_backend': 'LMCACHE_PIPELINED_BACKEND',
        'save_decode_cache': 'LMCACHE_SAVE_DECODE_CACHE'
    }
    
    # 设置环境变量
    for config_key, env_key in env_mapping.items():
        if config_key in config:
            value = config[config_key]
            if isinstance(value, bool):
                os.environ[env_key] = str(value).lower()
            else:
                os.environ[env_key] = str(value)
    
    # 设置默认值
    defaults = {
        'LMCACHE_CHUNK_SIZE': '256',
        'LMCACHE_LOCAL_DEVICE': 'cuda',
        'LMCACHE_MAX_LOCAL_CACHE_SIZE': '10.0',
        'LMCACHE_ENABLE_BLENDING': 'true',
        'LMCACHE_BLEND_RECOMPUTE_RATIO': '0.15',
        'LMCACHE_BLEND_MIN_TOKENS': '256',
        'LMCACHE_BLEND_SEPARATOR': '[LMCACHE_BLEND_SEP]',
        'LMCACHE_BLEND_ADD_SPECIAL_IN_PRECOMP': 'false',
        'LMCACHE_ENABLE_P2P': 'false',
        'LMCACHE_PIPELINED_BACKEND': 'false',
        'LMCACHE_SAVE_DECODE_CACHE': 'false'
    }
    
    for env_key, default_value in defaults.items():
        if env_key not in os.environ:
            os.environ[env_key] = default_value
    
    print("LMCache environment variables set successfully")


def get_lmcache_config_from_env() -> Dict[str, Any]:
    """
    从环境变量获取LMCache配置
    
    Returns:
        包含LMCache配置的字典
    """
    
    config = {}
    
    # 从环境变量读取配置
    env_mapping = {
        'LMCACHE_CHUNK_SIZE': 'chunk_size',
        'LMCACHE_LOCAL_DEVICE': 'local_device',
        'LMCACHE_MAX_LOCAL_CACHE_SIZE': 'max_local_cache_size',
        'LMCACHE_ENABLE_BLENDING': 'enable_blending',
        'LMCACHE_BLEND_RECOMPUTE_RATIO': 'blend_recompute_ratio',
        'LMCACHE_BLEND_MIN_TOKENS': 'blend_min_tokens',
        'LMCACHE_BLEND_SEPARATOR': 'blend_separator',
        'LMCACHE_BLEND_ADD_SPECIAL_IN_PRECOMP': 'blend_add_special_in_precomp',
        'LMCACHE_ENABLE_P2P': 'enable_p2p',
        'LMCACHE_REMOTE_URL': 'remote_url',
        'LMCACHE_REMOTE_SERDE': 'remote_serde',
        'LMCACHE_PIPELINED_BACKEND': 'pipelined_backend',
        'LMCACHE_SAVE_DECODE_CACHE': 'save_decode_cache'
    }
    
    for env_key, config_key in env_mapping.items():
        if env_key in os.environ:
            value = os.environ[env_key]
            
            # 类型转换
            if config_key in ['chunk_size', 'blend_min_tokens']:
                config[config_key] = int(value)
            elif config_key in ['max_local_cache_size', 'blend_recompute_ratio']:
                config[config_key] = float(value)
            elif config_key in ['enable_blending', 'blend_add_special_in_precomp', 'enable_p2p', 'pipelined_backend', 'save_decode_cache']:
                config[config_key] = value.lower() == 'true'
            else:
                config[config_key] = value
    
    return config


def create_lmcache_config_for_mmrag(args) -> Dict[str, Any]:
    """
    为MMRAG创建LMCache配置
    
    Args:
        args: MMRAG的参数对象
        
    Returns:
        包含LMCache配置的字典
    """
    
    config = {
        'chunk_size': getattr(args, 'chunk_size', 256),
        'local_device': 'cuda',
        'max_local_cache_size': getattr(args, 'max_cache_size', 10.0),
        'enable_blending': getattr(args, 'enable_blending', True),
        'blend_recompute_ratio': getattr(args, 'blend_recompute_ratio', 0.15),
        'blend_min_tokens': getattr(args, 'blend_min_tokens', 256),
        'blend_separator': '[LMCACHE_BLEND_SEP]',
        'blend_add_special_in_precomp': False,
        'enable_p2p': False,
        'pipelined_backend': False,
        'save_decode_cache': False
    }
    
    return config


def setup_lmcache_for_mmrag(args):
    """
    为MMRAG设置LMCache环境
    
    Args:
        args: MMRAG的参数对象
    """
    
    # 创建配置
    config = create_lmcache_config_for_mmrag(args)
    
    # 设置环境变量
    setup_lmcache_environment(config)
    
    print(f"LMCache environment setup completed with config: {config}")
    
    return config
