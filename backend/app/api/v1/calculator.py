import json
import os

import fastapi
from app.config import settings
from app.core.calculate_repository import CalculateRepository, OptimizationStrategyType, NetworkTopologyType
from app.models.calculator_input import Gpu, Model, Network, TrainningConfig, OptimalConfig
from app.models.calculator_input import OtherConfig, InputConfig
from app.models.calculator_result import Parameter, RecommendedConfig, MemoryUsage, \
    Computation, Communication, Timeline, TotalTime
from fastapi import Body, UploadFile, File
from fastapi.responses import FileResponse
from fastapi import HTTPException

router = fastapi.APIRouter()


@router.get("/gpu")
def gpu_list():
    return settings.GPU_LIST

@router.get("/network")
def get_network():
    return {
        "network_bandwidth": 0,
        "network_topology": [type.value for type in NetworkTopologyType]
    }

@router.get("/model")
def model_list():
    return settings.MODEL_LIST

@router.get("/optimization_strategies")
def get_optimization_strategies():
    return [strategy.value for strategy in OptimizationStrategyType]

@router.get("/datatype")
def get_gpu_datatype(gpu_name: str):
    try:
        # 构建文件路径 - 从当前文件位置向上查找到calculon根目录，然后进入systems
        current_dir = os.path.dirname(__file__)  # backend/app/api/v1/
        backend_dir = os.path.dirname(current_dir)  # backend/app/api/
        app_dir = os.path.dirname(backend_dir)  # backend/app/
        backend_root = os.path.dirname(app_dir)  # backend/
        calculon_root = os.path.dirname(backend_root)  # calculon根目录
        systems_dir = os.path.join(calculon_root, "systems")
        gpu_file_path = os.path.join(systems_dir, f"{gpu_name}.json")
        
        # 检查文件是否存在
        if not os.path.exists(gpu_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"GPU配置文件 {gpu_file_path} 未找到"
            )
        
        # 读取JSON文件
        with open(gpu_file_path, 'r', encoding='utf-8') as f:
            gpu_config = json.load(f)
        
        # 检查是否包含matrix和vector字段
        if "matrix" not in gpu_config or "vector" not in gpu_config:
            raise HTTPException(
                status_code=400,
                detail=f"GPU配置文件 {gpu_name}.json 格式错误：缺少matrix或vector字段"
            )
        
        # 获取matrix和vector中共同支持的数据类型
        matrix_datatypes = set(gpu_config["matrix"].keys())
        vector_datatypes = set(gpu_config["vector"].keys())
        
        # 找到交集
        common_datatypes = list(matrix_datatypes.intersection(vector_datatypes))
        
        if not common_datatypes:
            raise HTTPException(
                status_code=400,
                detail=f"GPU {gpu_name} 在matrix和vector中没有共同支持的数据类型"
            )
        
        return {
            "gpu_name": gpu_name,
            "datatypes": sorted(common_datatypes)
        }
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"GPU配置文件 {gpu_name}.json 格式错误：不是有效的JSON文件"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"读取GPU配置时发生错误: {str(e)}"
        )

@router.post("/parameter_metrics")
def calculate_params(model: Model):
    cr = CalculateRepository()
    params = cr.parameter_metrics(model)
    return params


@router.post("/recommended_tensor")
def recommended_tensor(cluster: Gpu, model: Model):
    cr = CalculateRepository()
    recomended_tensor_parallel_degree = cr.recommended_tensor(cluster, model)
    return recomended_tensor_parallel_degree


@router.post("/recommended_pipeline")
def recommended_pipeline(cluster: Gpu,
                         model: Model,
                         optimization_strategy: str = Body("Full recomputation"),
                         tensor_parallel_degree: int = Body(...)):
    cr = CalculateRepository()
    recomended_pipeline_parallel_degree = cr.recommended_pipeline(cluster, model, optimization_strategy,
                                                                  tensor_parallel_degree)
    return recomended_pipeline_parallel_degree


@router.post("/recommended_microbatch")
def recommended_microbatch(model: Model,
                           pipeline_parallel_degree: int = Body(...)):
    cr = CalculateRepository()
    recommended_config = cr.recommended_microbatch(model, pipeline_parallel_degree)
    return recommended_config


@router.post("/calculate")
def create_calculator(gpu: Gpu,
                      network: Network,
                      model: Model,
                      trainning_config: TrainningConfig):
    cr = CalculateRepository()
    res = cr.calculate(gpu, network, model, trainning_config)
    if res.get("status") == "error":
        raise HTTPException(
            status_code=400,  # 客户端参数错误
            detail=res["error"]
        )
    return res


@router.post("/optimal")
def create_optimal(gpu: Gpu,
                    network: Network,
                    model: Model,
                    optimal_config: OptimalConfig):
    cr = CalculateRepository()
    res = cr.optimal(gpu, network, model, optimal_config)
    if res.get("status") == "error":
        raise HTTPException(
            status_code=400,  # 客户端参数错误
            detail=res["error"]
        )
    return res


@router.post("/download")
def create_downloader(cluster: Gpu,
                      model: Model,
                      other_config: OtherConfig,
                      input_config: InputConfig,
                      parameter: Parameter,
                      recommended_config: RecommendedConfig,
                      memory_usage: MemoryUsage,
                      computation: Computation,
                      communication: Communication,
                      timeline: Timeline,
                      total_time: TotalTime):
    cr = CalculateRepository()
    file = cr.write_result_to_file(cluster, model, other_config, input_config, parameter, recommended_config,
                                   memory_usage,
                                   computation,
                                   communication, timeline, total_time)
    return FileResponse(file, filename="calculator.xlsx")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()  # 读取文件内容
    cr = CalculateRepository()
    tl, tt, other_config = cr.read_file_to_timeline(content)
    # 将tl和tt合并为一个字典
    result_dict = {"timeline": tl.dict(), "total_time": tt.dict(), "other_config": other_config.dict()}
    # 返回JSON格式的结果
    return result_dict


@router.post("/download_result_model")
def download_template():
    return FileResponse(settings.CALCULATOR_RESULT_TEMPLATE, filename="template.xlsx")
