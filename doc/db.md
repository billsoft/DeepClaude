# AI对话记录系统数据库设计优化 - 增加用户认证与权限管理

您提出了非常重要的一点。确实，在现代应用中，JWT认证和权限角色管理是不可或缺的功能。我将对之前的设计进行补充，添加相关表和字段来支持这些特性。

## 权限与认证相关表设计

除了之前设计的五张表外，我们需要新增三张表来支持完整的权限管理：

1. 角色表(roles)
2. 权限表(permissions)
3. 角色-权限关联表(role_permissions)

同时，用户表需要增加几个与认证相关的字段。

## 修改后的MySQL建表脚本

```sql
-- -----------------------------------------------------
-- 创建数据库 deepsysai (如果不存在)
-- -----------------------------------------------------
CREATE DATABASE IF NOT EXISTS `deepsysai` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `deepsysai`;

-- -----------------------------------------------------
-- 角色表
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `roles` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '角色ID，主键',
  `name` VARCHAR(50) NOT NULL COMMENT '角色名称',
  `description` VARCHAR(200) NULL COMMENT '角色描述',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE INDEX `role_name_UNIQUE` (`name` ASC)
) ENGINE = InnoDB COMMENT = '用户角色表';

-- -----------------------------------------------------
-- 权限表
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `permissions` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '权限ID，主键',
  `name` VARCHAR(100) NOT NULL COMMENT '权限名称',
  `code` VARCHAR(100) NOT NULL COMMENT '权限代码，用于程序识别',
  `description` VARCHAR(200) NULL COMMENT '权限描述',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE INDEX `permission_code_UNIQUE` (`code` ASC)
) ENGINE = InnoDB COMMENT = '权限表';

-- -----------------------------------------------------
-- 角色-权限关联表
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `role_permissions` (
  `role_id` INT UNSIGNED NOT NULL COMMENT '角色ID',
  `permission_id` INT UNSIGNED NOT NULL COMMENT '权限ID',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`role_id`, `permission_id`),
  INDEX `idx_permission_id` (`permission_id` ASC),
  CONSTRAINT `fk_roleperm_role`
    FOREIGN KEY (`role_id`)
    REFERENCES `roles` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_roleperm_permission`
    FOREIGN KEY (`permission_id`)
    REFERENCES `permissions` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '角色-权限关联表';

-- -----------------------------------------------------
-- 用户表 (修改后增加了JWT与角色相关字段)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `users` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户ID，主键',
  `username` VARCHAR(50) NOT NULL COMMENT '用户名',
  `password` VARCHAR(255) NOT NULL COMMENT '密码（加密存储）',
  `email` VARCHAR(100) NULL COMMENT '用户邮箱',
  `real_name` VARCHAR(50) NULL COMMENT '用户真实姓名',
  `phone` VARCHAR(20) NULL COMMENT '联系电话',
  `role_id` INT UNSIGNED NOT NULL COMMENT '角色ID，外键',
  `refresh_token` VARCHAR(500) NULL COMMENT 'JWT刷新令牌',
  `token_expire_time` DATETIME NULL COMMENT '令牌过期时间',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `last_login` DATETIME NULL COMMENT '最后登录时间',
  `status` TINYINT NOT NULL DEFAULT 1 COMMENT '状态：1-正常，0-禁用',
  `avatar` VARCHAR(255) NULL COMMENT '用户头像URL',
  `login_ip` VARCHAR(50) NULL COMMENT '最后登录IP',
  PRIMARY KEY (`id`),
  UNIQUE INDEX `username_UNIQUE` (`username` ASC),
  UNIQUE INDEX `email_UNIQUE` (`email` ASC),
  INDEX `idx_role_id` (`role_id` ASC),
  CONSTRAINT `fk_user_role`
    FOREIGN KEY (`role_id`)
    REFERENCES `roles` (`id`)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '用户信息表';

-- -----------------------------------------------------
-- 分类表 (保持不变)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `categories` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '分类ID，主键',
  `name` VARCHAR(100) NOT NULL COMMENT '分类名称',
  `parent_id` INT UNSIGNED NULL COMMENT '父分类ID，为空表示顶级分类',
  `description` VARCHAR(500) NULL COMMENT '分类描述',
  `sort_order` INT NOT NULL DEFAULT 0 COMMENT '排序顺序',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  INDEX `idx_parent_id` (`parent_id` ASC),
  CONSTRAINT `fk_category_parent`
    FOREIGN KEY (`parent_id`)
    REFERENCES `categories` (`id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '对话和知识的分类表';

-- -----------------------------------------------------
-- 对话列表表 (保持不变)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `conversation_lists` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '对话列表ID，主键',
  `user_id` INT UNSIGNED NOT NULL COMMENT '用户ID，外键',
  `title` VARCHAR(200) NULL COMMENT '对话标题，可自动生成或用户自定义',
  `category_id` INT UNSIGNED NULL COMMENT '分类ID，外键',
  `satisfaction` ENUM('satisfied', 'neutral', 'unsatisfied') NULL COMMENT '用户满意度评价',
  `feedback` TEXT NULL COMMENT '用户反馈内容',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `is_completed` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已完成：0-进行中，1-已完成',
  PRIMARY KEY (`id`),
  INDEX `idx_user_id` (`user_id` ASC),
  INDEX `idx_category_id` (`category_id` ASC),
  CONSTRAINT `fk_conversation_user`
    FOREIGN KEY (`user_id`)
    REFERENCES `users` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_conversation_category`
    FOREIGN KEY (`category_id`)
    REFERENCES `categories` (`id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '对话列表表，记录完整对话会话';

-- -----------------------------------------------------
-- 历史对话表 (保持不变)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `conversation_history` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '历史记录ID，主键',
  `conversation_id` INT UNSIGNED NOT NULL COMMENT '所属对话列表ID，外键',
  `user_id` INT UNSIGNED NOT NULL COMMENT '用户ID，外键',
  `role` ENUM('user', 'ai') NOT NULL COMMENT '发言角色：用户或AI',
  `content` TEXT NOT NULL COMMENT '对话内容',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `is_error` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否包含错误：0-正常，1-错误',
  `is_duplicate` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否重复内容：0-不是，1-是',
  `tokens` INT UNSIGNED NULL COMMENT 'Token数量，用于计算资源使用',
  `model_name` VARCHAR(100) NULL COMMENT '使用的AI模型名称',
  PRIMARY KEY (`id`),
  INDEX `idx_conversation_id` (`conversation_id` ASC),
  INDEX `idx_user_id` (`user_id` ASC),
  INDEX `idx_create_time` (`create_time` ASC),
  CONSTRAINT `fk_history_conversation`
    FOREIGN KEY (`conversation_id`)
    REFERENCES `conversation_lists` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_history_user`
    FOREIGN KEY (`user_id`)
    REFERENCES `users` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '历史对话表，记录具体问答内容';

-- -----------------------------------------------------
-- 知识库表 (保持不变)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_base` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '知识条目ID，主键',
  `question` VARCHAR(500) NOT NULL COMMENT '标准问题',
  `answer` TEXT NOT NULL COMMENT '标准答案',
  `source_conversation_id` INT UNSIGNED NULL COMMENT '来源对话ID，可为空',
  `category_id` INT UNSIGNED NULL COMMENT '分类ID，外键',
  `keywords` VARCHAR(500) NULL COMMENT '关键词，用于检索',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `creator_id` INT UNSIGNED NULL COMMENT '创建者ID，可能是自动提取或人工创建',
  `status` TINYINT NOT NULL DEFAULT 1 COMMENT '状态：1-启用，0-禁用',
  `confidence_score` FLOAT NULL COMMENT '置信度分数，表示该知识条目的可靠性',
  PRIMARY KEY (`id`),
  INDEX `idx_category_id` (`category_id` ASC),
  INDEX `idx_source_conversation` (`source_conversation_id` ASC),
  FULLTEXT INDEX `ft_question_answer` (`question`, `answer`),
  INDEX `idx_keywords` (`keywords`(255)),
  CONSTRAINT `fk_knowledge_category`
    FOREIGN KEY (`category_id`)
    REFERENCES `categories` (`id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
  CONSTRAINT `fk_knowledge_conversation`
    FOREIGN KEY (`source_conversation_id`)
    REFERENCES `conversation_lists` (`id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
) ENGINE = InnoDB COMMENT = '知识库表，存储标准问答样板';

-- -----------------------------------------------------
-- 初始化基本角色
-- -----------------------------------------------------
INSERT INTO `roles` (`name`, `description`) VALUES
('admin', '系统管理员，拥有所有权限'),
('manager', '管理人员，可查看所有对话和管理知识库'),
('user', '普通用户，可进行AI对话');
```

## 权限与认证相关说明

### 1. 角色与权限设计
- **RBAC模型**: 采用基于角色的访问控制模型，用户->角色->权限的三级结构
- **基础角色**: 默认提供管理员、管理人员和普通用户三种角色
- **灵活配置**: 可以为不同角色分配不同的权限，支持权限的精细化管理

### 2. JWT认证相关字段
- **refresh_token**: 用于刷新访问令牌，延长登录有效期
- **token_expire_time**: 明确记录令牌过期时间
- **last_login & login_ip**: 记录用户的登录情况，有助于安全审计

### 3. 用户表扩展
- 增加了角色关联，建立用户与权限之间的桥梁
- 增加了头像字段，提升用户体验
- 补充了安全相关字段，如登录IP等

### 4. 权限预设
在实际应用中，可以预设以下权限类型：
- 用户管理权限 (user:create, user:update, user:delete, user:view)
- 对话管理权限 (conversation:create, conversation:view, conversation:delete)
- 知识库管理权限 (knowledge:create, knowledge:update, knowledge:delete)
- 系统管理权限 (system:config, system:log, system:backup)

