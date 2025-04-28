CREATE TABLE `your_table_name` (
  `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
  `task_id` VARCHAR(100) NOT NULL DEFAULT '' COMMENT '任务ID',
  `uid` VARCHAR(100) NOT NULL DEFAULT '0' COMMENT '用户ID',
  `source` VARCHAR(20) NOT NULL DEFAULT 'miaobi' COMMENT '平台-user,miaobi',
  `status` VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT '任务状态 pending:待处理, processing:处理中, completed:已完成, failed:失败',
  `payload` JSON NULL COMMENT '任务请求参数',
  `result` JSON NULL COMMENT '任务处理结果',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  KEY `idx_task_id` (`task_id`),
  KEY `idx_source` (`source`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='滚动视频任务表';
