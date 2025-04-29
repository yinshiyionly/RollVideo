CREATE TABLE `roll_video_task` (
  `id` int unsigned NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` varchar(100) NOT NULL DEFAULT '' COMMENT '任务ID',
  `uid` int unsigned NOT NULL DEFAULT '0' COMMENT '用户ID',
  `source` varchar(20) NOT NULL DEFAULT '' COMMENT '来源',
  `task_state` varchar(20) NOT NULL DEFAULT 'pending' COMMENT '任务状态 pending:待处理, processing:处理中, completed:已完成, failed:失败',
  `payload` json DEFAULT NULL COMMENT '任务请求参数',
  `result` json DEFAULT NULL COMMENT '任务结果',
  `status` tinyint unsigned NOT NULL DEFAULT '1' COMMENT '状态 1-正常 2-删除',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx-task_id` (`task_id`),
  KEY `idx-source` (`source`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='滚动视频任务表';