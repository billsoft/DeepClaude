-- 为conversation_history表添加reasoning列
ALTER TABLE `conversation_history` 
ADD COLUMN `reasoning` TEXT NULL COMMENT '思考过程内容' AFTER `model_name`;

-- 确认已添加
SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'conversation_history' AND COLUMN_NAME = 'reasoning'; 