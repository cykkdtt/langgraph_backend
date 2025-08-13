"""
并行执行器

提供并行任务执行和管理功能。
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .workflow_types import (
    ParallelTaskConfig, WorkflowStep, WorkflowExecution,
    StepStatus, ExecutionMode
)


class ParallelTask:
    """并行任务"""
    
    def __init__(
        self,
        task_id: str,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.timeout = timeout
        self.retry_count = retry_count
        self.metadata = metadata or {}
        
        # 执行状态
        self.status = StepStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.attempts = 0
    
    async def execute(self) -> Any:
        """执行任务"""
        self.start_time = datetime.now()
        self.status = StepStatus.RUNNING
        self.attempts += 1
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                # 异步函数
                if self.timeout:
                    self.result = await asyncio.wait_for(
                        self.func(*self.args, **self.kwargs),
                        timeout=self.timeout
                    )
                else:
                    self.result = await self.func(*self.args, **self.kwargs)
            else:
                # 同步函数
                loop = asyncio.get_event_loop()
                if self.timeout:
                    self.result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, 
                            lambda: self.func(*self.args, **self.kwargs)
                        ),
                        timeout=self.timeout
                    )
                else:
                    self.result = await loop.run_in_executor(
                        None, 
                        lambda: self.func(*self.args, **self.kwargs)
                    )
            
            self.status = StepStatus.COMPLETED
            self.end_time = datetime.now()
            return self.result
        
        except asyncio.TimeoutError:
            self.status = StepStatus.FAILED
            self.error = TimeoutError(f"任务 {self.name} 执行超时")
            self.end_time = datetime.now()
            raise self.error
        
        except Exception as e:
            self.status = StepStatus.FAILED
            self.error = e
            self.end_time = datetime.now()
            
            # 检查是否需要重试
            if self.attempts <= self.retry_count:
                self.status = StepStatus.PENDING
                await asyncio.sleep(1)  # 重试延迟
                return await self.execute()
            
            raise e
    
    def get_duration(self) -> Optional[float]:
        """获取执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "attempts": self.attempts,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata
        }


class ParallelExecutor:
    """并行执行器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._running_tasks: Dict[str, ParallelTask] = {}
        self._completed_tasks: Dict[str, ParallelTask] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """注册任务处理器"""
        self._task_handlers[task_type] = handler
    
    async def execute_parallel_tasks(
        self,
        config: ParallelTaskConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行并行任务配置"""
        tasks = []
        
        # 创建任务
        for i, task_name in enumerate(config.tasks):
            task_id = f"{config.id}_{i}"
            
            # 获取任务处理器
            if task_name in self._task_handlers:
                handler = self._task_handlers[task_name]
            else:
                # 默认处理器
                handler = self._default_task_handler
            
            task = ParallelTask(
                task_id=task_id,
                name=task_name,
                func=handler,
                args=(task_name, context),
                timeout=config.timeout,
                metadata={"config_id": config.id, "task_index": i}
            )
            
            tasks.append(task)
        
        # 执行任务
        return await self.execute_tasks(
            tasks,
            max_concurrency=config.max_concurrency,
            wait_for_all=config.wait_for_all
        )
    
    async def execute_tasks(
        self,
        tasks: List[ParallelTask],
        max_concurrency: Optional[int] = None,
        wait_for_all: bool = True
    ) -> Dict[str, Any]:
        """执行任务列表"""
        if not tasks:
            return {"results": {}, "status": "completed", "summary": {}}
        
        # 限制并发数
        concurrency = min(
            max_concurrency or len(tasks),
            len(tasks),
            self.max_workers
        )
        
        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        
        async def execute_with_semaphore(task: ParallelTask) -> Tuple[str, Any]:
            async with semaphore:
                with self._lock:
                    self._running_tasks[task.task_id] = task
                
                try:
                    result = await task.execute()
                    return task.task_id, result
                finally:
                    with self._lock:
                        if task.task_id in self._running_tasks:
                            del self._running_tasks[task.task_id]
                        self._completed_tasks[task.task_id] = task
        
        # 创建协程
        coroutines = [execute_with_semaphore(task) for task in tasks]
        
        results = {}
        errors = {}
        completed_count = 0
        
        if wait_for_all:
            # 等待所有任务完成
            try:
                task_results = await asyncio.gather(*coroutines, return_exceptions=True)
                
                for i, result in enumerate(task_results):
                    task = tasks[i]
                    if isinstance(result, Exception):
                        errors[task.task_id] = str(result)
                    else:
                        task_id, task_result = result
                        results[task_id] = task_result
                        completed_count += 1
            
            except Exception as e:
                # 处理整体执行错误
                for task in tasks:
                    if task.task_id not in results and task.task_id not in errors:
                        errors[task.task_id] = str(e)
        
        else:
            # 不等待所有任务，收集已完成的结果
            done_tasks = []
            pending_tasks = coroutines
            
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for done_task in done:
                    try:
                        result = await done_task
                        task_id, task_result = result
                        results[task_id] = task_result
                        completed_count += 1
                    except Exception as e:
                        # 找到对应的任务
                        for task in tasks:
                            if task.task_id not in results and task.task_id not in errors:
                                errors[task.task_id] = str(e)
                                break
                
                # 如果有足够的成功结果，可以提前退出
                if completed_count >= len(tasks) // 2:  # 超过一半成功
                    # 取消剩余任务
                    for pending_task in pending_tasks:
                        pending_task.cancel()
                    break
        
        # 生成执行摘要
        summary = self._generate_execution_summary(tasks, results, errors)
        
        return {
            "results": results,
            "errors": errors,
            "status": "completed" if not errors else "partial",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_task_batch(
        self,
        task_batches: List[List[ParallelTask]],
        batch_delay: float = 0.0
    ) -> Dict[str, Any]:
        """分批执行任务"""
        all_results = {}
        all_errors = {}
        batch_summaries = []
        
        for i, batch in enumerate(task_batches):
            print(f"执行第 {i + 1} 批任务，共 {len(batch)} 个任务")
            
            batch_result = await self.execute_tasks(batch)
            
            all_results.update(batch_result["results"])
            all_errors.update(batch_result["errors"])
            batch_summaries.append({
                "batch_index": i,
                "summary": batch_result["summary"]
            })
            
            # 批次间延迟
            if batch_delay > 0 and i < len(task_batches) - 1:
                await asyncio.sleep(batch_delay)
        
        return {
            "results": all_results,
            "errors": all_errors,
            "status": "completed" if not all_errors else "partial",
            "batch_summaries": batch_summaries,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_running_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取正在运行的任务"""
        with self._lock:
            return {
                task_id: task.to_dict()
                for task_id, task in self._running_tasks.items()
            }
    
    def get_completed_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取已完成的任务"""
        with self._lock:
            return {
                task_id: task.to_dict()
                for task_id, task in self._completed_tasks.items()
            }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self._lock:
            if task_id in self._running_tasks:
                return self._running_tasks[task_id].to_dict()
            elif task_id in self._completed_tasks:
                return self._completed_tasks[task_id].to_dict()
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.status = StepStatus.CANCELLED
                return True
            return False
    
    def clear_completed_tasks(self) -> None:
        """清理已完成的任务"""
        with self._lock:
            self._completed_tasks.clear()
    
    def _generate_execution_summary(
        self,
        tasks: List[ParallelTask],
        results: Dict[str, Any],
        errors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成执行摘要"""
        total_tasks = len(tasks)
        successful_tasks = len(results)
        failed_tasks = len(errors)
        
        # 计算执行时间统计
        durations = []
        for task in tasks:
            duration = task.get_duration()
            if duration is not None:
                durations.append(duration)
        
        duration_stats = {}
        if durations:
            duration_stats = {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations)
            }
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "duration_stats": duration_stats,
            "task_details": [task.to_dict() for task in tasks]
        }
    
    async def _default_task_handler(self, task_name: str, context: Dict[str, Any]) -> Any:
        """默认任务处理器"""
        # 模拟任务执行
        await asyncio.sleep(0.1)
        return {
            "task_name": task_name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "context_keys": list(context.keys())
        }


# 导入 os 模块
import os