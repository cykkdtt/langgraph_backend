"""
LangMem提示词优化API路由

本模块提供LangMem提示词优化功能的REST API接口，包括：
- 收集用户反馈
- 优化单个智能体提示词
- 优化多智能体系统
- 获取优化历史
- 自动化优化管理
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field

from .prompt_optimizer import PromptOptimizer, FeedbackCollector, AutoOptimizationScheduler

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/optimization", tags=["prompt_optimization"])

# 数据模型
class FeedbackRequest(BaseModel):
    """用户反馈请求模型"""
    agent_type: str = Field(..., description="智能体类型")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    satisfaction_score: int = Field(..., ge=1, le=5, description="满意度评分(1-5)")
    feedback_text: Optional[str] = Field(None, description="反馈文本")
    improvement_suggestions: Optional[List[str]] = Field(None, description="改进建议")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")


class OptimizationRequest(BaseModel):
    """优化请求模型"""
    agent_type: str = Field(..., description="智能体类型")
    user_id: Optional[str] = Field(None, description="用户ID")
    optimization_strategy: str = Field("gradient", description="优化策略")
    max_iterations: int = Field(5, ge=1, le=20, description="最大迭代次数")
    context: Optional[Dict[str, Any]] = Field(None, description="优化上下文")


class MultiAgentOptimizationRequest(BaseModel):
    """多智能体优化请求模型"""
    agent_types: List[str] = Field(..., description="智能体类型列表")
    user_id: Optional[str] = Field(None, description="用户ID")
    optimization_strategy: str = Field("gradient", description="优化策略")
    max_iterations: int = Field(5, ge=1, le=20, description="最大迭代次数")
    context: Optional[Dict[str, Any]] = Field(None, description="优化上下文")


class OptimizationResponse(BaseModel):
    """优化响应模型"""
    success: bool
    optimization_id: str
    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    optimization_details: Dict[str, Any]
    timestamp: datetime


class OptimizationHistoryResponse(BaseModel):
    """优化历史响应模型"""
    optimizations: List[Dict[str, Any]]
    total_count: int
    average_improvement: float


class AutoOptimizationConfig(BaseModel):
    """自动优化配置模型"""
    enabled: bool = Field(True, description="是否启用自动优化")
    interval_hours: int = Field(24, ge=1, le=168, description="优化间隔(小时)")
    min_feedback_count: int = Field(10, ge=1, description="最小反馈数量")
    optimization_strategy: str = Field("gradient", description="优化策略")


# 依赖函数
async def get_prompt_optimizer(request: Request) -> PromptOptimizer:
    """获取提示词优化器实例"""
    if not hasattr(request.app.state, 'prompt_optimizer'):
        raise HTTPException(status_code=503, detail="提示词优化器未初始化")
    return request.app.state.prompt_optimizer


async def get_feedback_collector(request: Request) -> FeedbackCollector:
    """获取反馈收集器实例"""
    if not hasattr(request.app.state, 'feedback_collector'):
        raise HTTPException(status_code=503, detail="反馈收集器未初始化")
    return request.app.state.feedback_collector


async def get_auto_scheduler(request: Request) -> AutoOptimizationScheduler:
    """获取自动优化调度器实例"""
    if not hasattr(request.app.state, 'auto_scheduler'):
        raise HTTPException(status_code=503, detail="自动优化调度器未初始化")
    return request.app.state.auto_scheduler


# API路由
@router.post("/feedback", summary="提交用户反馈")
async def submit_feedback(feedback: FeedbackRequest, collector: FeedbackCollector = Depends(get_feedback_collector)):
    """提交用户反馈"""
    try:
        feedback_id = await collector.collect_feedback(
            agent_type=feedback.agent_type,
            user_id=feedback.user_id,
            session_id=feedback.session_id,
            satisfaction_score=feedback.satisfaction_score,
            feedback_text=feedback.feedback_text,
            improvement_suggestions=feedback.improvement_suggestions or [],
            context=feedback.context or {}
        )
        
        logger.info(f"收集到用户反馈: {feedback_id}")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "反馈提交成功",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.post("/optimize/single", response_model=OptimizationResponse, summary="优化单个智能体提示词")
async def optimize_single_agent(
    request: OptimizationRequest, 
    background_tasks: BackgroundTasks,
    optimizer: PromptOptimizer = Depends(get_prompt_optimizer)
):
    """优化单个智能体的提示词"""
    try:
        # 执行优化
        result = await optimizer.optimize_agent_prompt(
            agent_type=request.agent_type,
            user_id=request.user_id,
            strategy=request.optimization_strategy,
            max_iterations=request.max_iterations,
            context=request.context or {}
        )
        
        logger.info(f"单智能体优化完成: {request.agent_type}")
        
        return OptimizationResponse(
            success=True,
            optimization_id=result["optimization_id"],
            original_prompt=result["original_prompt"],
            optimized_prompt=result["optimized_prompt"],
            improvement_score=result["improvement_score"],
            optimization_details=result["details"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"单智能体优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")


@router.post("/optimize/multi", summary="优化多智能体系统")
async def optimize_multi_agent(
    request: MultiAgentOptimizationRequest, 
    background_tasks: BackgroundTasks,
    optimizer: PromptOptimizer = Depends(get_prompt_optimizer)
):
    """优化多智能体系统的协同提示词"""
    try:
        # 执行多智能体优化
        result = await optimizer.optimize_multi_agent_system(
            agent_types=request.agent_types,
            user_id=request.user_id,
            strategy=request.optimization_strategy,
            max_iterations=request.max_iterations,
            context=request.context or {}
        )
        
        logger.info(f"多智能体优化完成: {request.agent_types}")
        
        return {
            "success": True,
            "optimization_id": result["optimization_id"],
            "optimized_agents": result["optimized_agents"],
            "overall_improvement": result["overall_improvement"],
            "optimization_details": result["details"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"多智能体优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"多智能体优化失败: {str(e)}")


@router.get("/history/{agent_type}", response_model=OptimizationHistoryResponse, summary="获取优化历史")
async def get_optimization_history(
    agent_type: str,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    optimizer: PromptOptimizer = Depends(get_prompt_optimizer)
):
    """获取指定智能体的优化历史"""
    try:
        # 构建查询条件
        query_filter = {"agent_type": agent_type}
        if user_id:
            query_filter["user_id"] = user_id
        
        # 获取优化历史
        history = await optimizer.get_optimization_history(
            filter_dict=query_filter,
            limit=limit,
            offset=offset
        )
        
        # 计算平均改进分数
        improvements = [h.get("improvement_score", 0) for h in history]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        
        return OptimizationHistoryResponse(
            optimizations=history,
            total_count=len(history),
            average_improvement=avg_improvement
        )
        
    except Exception as e:
        logger.error(f"获取优化历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取优化历史失败: {str(e)}")


@router.get("/feedback/{agent_type}", summary="获取反馈统计")
async def get_feedback_stats(
    agent_type: str,
    user_id: Optional[str] = None,
    days: int = 30,
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """获取指定智能体的反馈统计"""
    try:
        stats = await collector.get_feedback_stats(
            agent_type=agent_type,
            user_id=user_id,
            days=days
        )
        
        return {
            "success": True,
            "agent_type": agent_type,
            "stats": stats,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"获取反馈统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取反馈统计失败: {str(e)}")


@router.post("/auto/config", summary="配置自动优化")
async def configure_auto_optimization(
    config: AutoOptimizationConfig,
    scheduler: AutoOptimizationScheduler = Depends(get_auto_scheduler)
):
    """配置自动优化参数"""
    try:
        if config.enabled:
            await scheduler.start_auto_optimization(
                interval_hours=config.interval_hours,
                min_feedback_count=config.min_feedback_count,
                optimization_strategy=config.optimization_strategy
            )
            message = "自动优化已启动"
        else:
            await scheduler.stop_auto_optimization()
            message = "自动优化已停止"
        
        logger.info(f"自动优化配置更新: {config}")
        
        return {
            "success": True,
            "message": message,
            "config": config.dict(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"配置自动优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置自动优化失败: {str(e)}")


@router.get("/auto/status", summary="获取自动优化状态")
async def get_auto_optimization_status(scheduler: AutoOptimizationScheduler = Depends(get_auto_scheduler)):
    """获取自动优化状态"""
    try:
        status = await scheduler.get_status()
        
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"获取自动优化状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取自动优化状态失败: {str(e)}")


@router.post("/auto/trigger", summary="手动触发自动优化")
async def trigger_auto_optimization(
    background_tasks: BackgroundTasks,
    scheduler: AutoOptimizationScheduler = Depends(get_auto_scheduler)
):
    """手动触发一次自动优化"""
    try:
        # 在后台执行优化
        background_tasks.add_task(scheduler.run_optimization_cycle)
        
        logger.info("手动触发自动优化")
        
        return {
            "success": True,
            "message": "自动优化已在后台启动",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"触发自动优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发自动优化失败: {str(e)}")


@router.get("/strategies", summary="获取可用的优化策略")
async def get_optimization_strategies():
    """获取可用的优化策略列表"""
    return {
        "strategies": [
            {
                "name": "gradient",
                "description": "基于梯度的优化，通过分析反馈梯度调整提示词",
                "suitable_for": ["单智能体", "快速优化"]
            },
            {
                "name": "prompt_memory",
                "description": "基于记忆的优化，利用历史成功案例优化提示词",
                "suitable_for": ["多智能体", "长期优化"]
            },
            {
                "name": "metaprompt",
                "description": "元提示词优化，通过元学习改进提示词结构",
                "suitable_for": ["复杂任务", "结构化优化"]
            }
        ]
    }


# 健康检查
@router.get("/health", summary="提示词优化模块健康检查")
async def health_check(
    optimizer: PromptOptimizer = Depends(get_prompt_optimizer),
    collector: FeedbackCollector = Depends(get_feedback_collector),
    scheduler: AutoOptimizationScheduler = Depends(get_auto_scheduler)
):
    """提示词优化模块健康检查"""
    try:
        # 检查各组件状态
        optimizer_status = await optimizer.health_check()
        collector_status = await collector.health_check()
        scheduler_status = await scheduler.get_status()
        
        return {
            "status": "healthy",
            "components": {
                "prompt_optimizer": optimizer_status,
                "feedback_collector": collector_status,
                "auto_scheduler": scheduler_status
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }