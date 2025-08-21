"""附件仓储模块

提供附件相关的数据访问接口和业务逻辑。
"""

import logging
import os
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import Session, selectinload

from ..models.database import Attachment, AttachmentType
from ..models.api import AttachmentCreate, AttachmentUpdate, AttachmentStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class AttachmentRepository(CRUDRepository[Attachment]):
    """附件仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(Attachment, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(Attachment.message)
        )
    
    def create_attachment(
        self, 
        message_id: int,
        filename: str,
        file_path: str,
        file_size: int,
        mime_type: str,
        attachment_type: AttachmentType = AttachmentType.FILE,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> Attachment:
        """创建附件"""
        attachment_data = {
            "message_id": message_id,
            "filename": filename,
            "file_path": file_path,
            "file_size": file_size,
            "mime_type": mime_type,
            "attachment_type": attachment_type,
            "metadata": metadata or {},
            "is_active": True
        }
        
        return self.create(attachment_data, session)
    
    def get_message_attachments(
        self, 
        message_id: int,
        attachment_type: Optional[AttachmentType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """获取消息的附件列表"""
        filters = QueryFilter().and_(
            QueryFilter().eq("message_id", message_id),
            QueryFilter().eq("is_active", True)
        )
        
        if attachment_type:
            filters = filters.and_(QueryFilter().eq("attachment_type", attachment_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_attachments_by_type(
        self, 
        attachment_type: AttachmentType,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """根据类型获取附件"""
        filters = QueryFilter().and_(
            QueryFilter().eq("attachment_type", attachment_type),
            QueryFilter().eq("is_active", True)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_attachments_by_mime_type(
        self, 
        mime_type: str,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """根据MIME类型获取附件"""
        filters = QueryFilter().and_(
            QueryFilter().eq("mime_type", mime_type),
            QueryFilter().eq("is_active", True)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def search_attachments(
        self, 
        query: str,
        attachment_type: Optional[AttachmentType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """搜索附件"""
        filters = QueryFilter().and_(
            QueryFilter().or_(
                QueryFilter().ilike("filename", f"%{query}%"),
                QueryFilter().ilike("mime_type", f"%{query}%")
            ),
            QueryFilter().eq("is_active", True)
        )
        
        if attachment_type:
            filters = filters.and_(QueryFilter().eq("attachment_type", attachment_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_large_attachments(
        self, 
        min_size_mb: float = 10.0,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """获取大文件附件"""
        min_size_bytes = int(min_size_mb * 1024 * 1024)  # 转换为字节
        
        filters = QueryFilter().and_(
            QueryFilter().gte("file_size", min_size_bytes),
            QueryFilter().eq("is_active", True)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("file_size", "desc")],
            session=session
        )
    
    def get_recent_attachments(
        self, 
        hours: int = 24,
        attachment_type: Optional[AttachmentType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """获取最近的附件"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        filters = QueryFilter().and_(
            QueryFilter().gte("created_at", since_time),
            QueryFilter().eq("is_active", True)
        )
        
        if attachment_type:
            filters = filters.and_(QueryFilter().eq("attachment_type", attachment_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_attachments_by_date_range(
        self, 
        start_date: datetime,
        end_date: datetime,
        attachment_type: Optional[AttachmentType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """根据日期范围获取附件"""
        filters = QueryFilter().and_(
            QueryFilter().gte("created_at", start_date),
            QueryFilter().lte("created_at", end_date),
            QueryFilter().eq("is_active", True)
        )
        
        if attachment_type:
            filters = filters.and_(QueryFilter().eq("attachment_type", attachment_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def update_attachment_metadata(
        self, 
        attachment_id: int,
        metadata: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[Attachment]:
        """更新附件元数据"""
        try:
            attachment = self.get(attachment_id, session)
            if not attachment:
                return None
            
            # 合并元数据
            updated_metadata = attachment.metadata.copy() if attachment.metadata else {}
            updated_metadata.update(metadata)
            
            return self.update(
                attachment_id,
                {"metadata": updated_metadata},
                session
            )
            
        except Exception as e:
            logger.error(f"Error updating attachment metadata: {e}")
            return None
    
    def mark_attachment_as_processed(
        self, 
        attachment_id: int,
        processing_result: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[Attachment]:
        """标记附件为已处理"""
        metadata = {
            "processed": True,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_result": processing_result
        }
        
        return self.update_attachment_metadata(attachment_id, metadata, session)
    
    def get_unprocessed_attachments(
        self, 
        attachment_type: Optional[AttachmentType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """获取未处理的附件"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            query = db_session.query(Attachment)
            query = query.filter(
                Attachment.is_active == True,
                or_(
                    Attachment.metadata.is_(None),
                    Attachment.metadata.op('->>')('processed').is_(None),
                    Attachment.metadata.op('->>')('processed') == 'false'
                )
            )
            
            if attachment_type:
                query = query.filter(Attachment.attachment_type == attachment_type)
            
            query = query.order_by(desc(Attachment.created_at))
            query = query.offset(skip).limit(limit)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error getting unprocessed attachments: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_attachment_statistics(
        self, 
        message_id: Optional[int] = None,
        days: int = 30,
        session: Optional[Session] = None
    ) -> AttachmentStats:
        """获取附件统计信息"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            base_filters = QueryFilter().and_(
                QueryFilter().gte("created_at", start_date),
                QueryFilter().eq("is_active", True)
            )
            
            if message_id:
                base_filters = base_filters.and_(QueryFilter().eq("message_id", message_id))
            
            # 总附件数
            total_attachments = self.count(filters=base_filters, session=session)
            
            # 今日附件数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_attachments = self.count(filters=today_filters, session=session)
            
            # 按类型统计
            image_filters = base_filters.and_(QueryFilter().eq("attachment_type", AttachmentType.IMAGE))
            image_count = self.count(filters=image_filters, session=session)
            
            document_filters = base_filters.and_(QueryFilter().eq("attachment_type", AttachmentType.DOCUMENT))
            document_count = self.count(filters=document_filters, session=session)
            
            audio_filters = base_filters.and_(QueryFilter().eq("attachment_type", AttachmentType.AUDIO))
            audio_count = self.count(filters=audio_filters, session=session)
            
            video_filters = base_filters.and_(QueryFilter().eq("attachment_type", AttachmentType.VIDEO))
            video_count = self.count(filters=video_filters, session=session)
            
            file_filters = base_filters.and_(QueryFilter().eq("attachment_type", AttachmentType.FILE))
            file_count = self.count(filters=file_filters, session=session)
            
            # 计算总文件大小
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            try:
                query = db_session.query(func.sum(Attachment.file_size))
                query = query.filter(
                    Attachment.created_at >= start_date,
                    Attachment.is_active == True
                )
                
                if message_id:
                    query = query.filter(Attachment.message_id == message_id)
                
                total_size_result = query.scalar()
                total_size = total_size_result if total_size_result else 0
                
                # 计算平均文件大小
                if total_attachments > 0:
                    average_size = total_size / total_attachments
                else:
                    average_size = 0
                
            except Exception as e:
                logger.error(f"Error calculating file sizes: {e}")
                total_size = 0
                average_size = 0
            finally:
                if not session:
                    db_session.close()
            
            return AttachmentStats(
                total_attachments=total_attachments,
                today_attachments=today_attachments,
                image_count=image_count,
                document_count=document_count,
                audio_count=audio_count,
                video_count=video_count,
                file_count=file_count,
                total_size=total_size,
                average_size=average_size
            )
            
        except Exception as e:
            logger.error(f"Error getting attachment statistics: {e}")
            return AttachmentStats(
                total_attachments=0,
                today_attachments=0,
                image_count=0,
                document_count=0,
                audio_count=0,
                video_count=0,
                file_count=0,
                total_size=0,
                average_size=0
            )
    
    def cleanup_orphaned_attachments(
        self, 
        session: Optional[Session] = None
    ) -> int:
        """清理孤立的附件（没有关联消息的附件）"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查找孤立的附件
            from ..models.database import Message
            
            orphaned_query = db_session.query(Attachment)
            orphaned_query = orphaned_query.outerjoin(Message, Attachment.message_id == Message.id)
            orphaned_query = orphaned_query.filter(
                Message.id.is_(None),
                Attachment.is_active == True
            )
            
            orphaned_attachments = orphaned_query.all()
            
            # 软删除孤立的附件
            deleted_count = 0
            for attachment in orphaned_attachments:
                if self.soft_delete(attachment.id, db_session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} orphaned attachments")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned attachments: {e}")
            return 0
        finally:
            if not session:
                db_session.close()
    
    def cleanup_old_attachments(
        self, 
        days: int = 365,
        session: Optional[Session] = None
    ) -> int:
        """清理旧附件"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            filters = QueryFilter().and_(
                QueryFilter().lt("created_at", cutoff_date),
                QueryFilter().eq("is_active", True)
            )
            
            old_attachments = self.get_multi(
                filters=filters,
                limit=1000,
                session=session
            )
            
            # 软删除旧附件
            deleted_count = 0
            for attachment in old_attachments:
                if self.soft_delete(attachment.id, session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old attachments")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old attachments: {e}")
            return 0
    
    def get_file_extension_statistics(
        self, 
        days: int = 30,
        session: Optional[Session] = None
    ) -> Dict[str, int]:
        """获取文件扩展名统计"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查询文件扩展名统计
            query = db_session.query(
                func.lower(func.split_part(Attachment.filename, '.', -1)).label('extension'),
                func.count().label('count')
            )
            query = query.filter(
                Attachment.created_at >= start_date,
                Attachment.is_active == True,
                Attachment.filename.contains('.')
            )
            query = query.group_by('extension')
            query = query.order_by(desc('count'))
            
            results = query.all()
            
            return {result.extension: result.count for result in results}
            
        except Exception as e:
            logger.error(f"Error getting file extension statistics: {e}")
            return {}
        finally:
            if not session:
                db_session.close()
    
    def validate_file_path(
        self, 
        file_path: str
    ) -> bool:
        """验证文件路径是否存在"""
        try:
            return os.path.exists(file_path) and os.path.isfile(file_path)
        except Exception as e:
            logger.error(f"Error validating file path {file_path}: {e}")
            return False
    
    def get_broken_attachments(
        self, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Attachment]:
        """获取损坏的附件（文件路径不存在）"""
        try:
            attachments = self.get_multi(
                filters=QueryFilter().eq("is_active", True),
                skip=skip,
                limit=limit,
                session=session
            )
            
            broken_attachments = []
            for attachment in attachments:
                if not self.validate_file_path(attachment.file_path):
                    broken_attachments.append(attachment)
            
            return broken_attachments
            
        except Exception as e:
            logger.error(f"Error getting broken attachments: {e}")
            return []
    
    def get_duplicate_attachments(
        self, 
        session: Optional[Session] = None
    ) -> List[List[Attachment]]:
        """获取重复的附件（基于文件大小和文件名）"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查找具有相同文件名和大小的附件
            query = db_session.query(
                Attachment.filename,
                Attachment.file_size,
                func.count().label('count')
            )
            query = query.filter(Attachment.is_active == True)
            query = query.group_by(Attachment.filename, Attachment.file_size)
            query = query.having(func.count() > 1)
            
            duplicates_info = query.all()
            
            duplicate_groups = []
            for info in duplicates_info:
                # 获取具有相同文件名和大小的所有附件
                attachments = db_session.query(Attachment).filter(
                    Attachment.filename == info.filename,
                    Attachment.file_size == info.file_size,
                    Attachment.is_active == True
                ).all()
                
                duplicate_groups.append(attachments)
            
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error getting duplicate attachments: {e}")
            return []
        finally:
            if not session:
                db_session.close()