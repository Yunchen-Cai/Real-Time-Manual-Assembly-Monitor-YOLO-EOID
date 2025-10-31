CREATE TABLE Assembly_Sequence (
    step_id INT PRIMARY KEY AUTO_INCREMENT,
    step_description VARCHAR(255) NOT NULL,
    action_id INT NOT NULL,
    action_description VARCHAR(255) NOT NULL,
    object_id INT NOT NULL,
    object_description VARCHAR(255) NOT NULL
);


INSERT INTO Assembly_Sequence (step_description, action_id, action_description, object_id, object_description)
VALUES
( 'Install the display board', 1, 'Pressing with hands', 1, 'Overall display board' ),
( 'Install the three screws for the display panel', 9, 'Install with screw driver', 9, 'Display panel screws' ),
( 'Install the heating plate', 13, 'Holding with hand', 13, 'Heating plate' ),
( 'Install the two screws for the heating plate', 9, 'Install with screw driver', 21, 'Heating plate screws' ),
( 'Placing the heat sensor', 10, 'Place with hand', 22, 'Heat sensor' ),
( 'Install the two screws for the heat sensor', 9, 'Install with screw driver', 23, 'Heat sensor screws' ),
( 'Install the power connector', 25, 'Holding with hands', 25, 'Power connector' ),
( 'Install the base plate', 29, 'Cover with back cover', 29, 'Back cover' ),
( 'Install the inner pot', 57, 'Install the inner pot', 57, 'Inner pot' ),
( 'Install the steam valve', 61, 'Install the steam valve', 61, 'Steam valve' ),
( 'Install the upper inner lid', 65, 'Positioning with hand', 65, 'Upper inner lid' ),
( 'Install the handle', 69, 'Take with hand', 69, 'Handle' );


 # 查询所有装配步骤
SELECT * FROM Assembly_Sequence;
# 📌 查询特定零件的装配步骤
SELECT * FROM Assembly_Sequence WHERE object_description = 'Heating plate';
# 📌 查询使用螺丝刀安装的所有步骤
SELECT * FROM Assembly_Sequence WHERE action_description LIKE '%screw driver%';
# 📌 查询所有涉及"手"的操作
SELECT * FROM Assembly_Sequence WHERE action_description LIKE '%hand%';