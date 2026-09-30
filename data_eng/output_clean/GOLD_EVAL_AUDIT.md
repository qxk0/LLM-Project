# 人工 Gold Evaluation Seed 审计表 (GOLD_EVAL_AUDIT)

- gold_eval 总数: **78**(期望 78)
- 全部进入 test: True
- 按 intent: {'after_sale': 10, 'allergen': 10, 'delivery': 6, 'hours': 8, 'member': 5, 'ood': 10, 'price': 12, 'recommend': 12, 'store': 5}

## 0. 与 train / val / 旧 test 的多层去重结果(必须全为 0)

```text
gold_eval ∩ train: exact=0 base=0 family=0
gold_eval ∩ val: exact=0 base=0 family=0
gold_eval ∩ old_test: exact=0 base=0 family=0
gold_eval OOD family ∩ train/val/old_test: {'train': 0, 'val': 0, 'old_test': 0}
```

## 1. 最终 test 的 intent 覆盖

| intent | test 题数 | 其中 gold_eval |
|---|---|---|
| after_sale | 10 | 10 |
| allergen | 10 | 10 |
| customize | 30 | 0 |
| delivery | 11 | 6 |
| hours | 13 | 8 |
| member | 15 | 5 |
| ood | 60 | 10 |
| price | 12 | 12 |
| recommend | 12 | 12 |
| store | 10 | 5 |

- test 未覆盖的 intent:无
- 上一轮 clean test 保留情况:{'previous_test_rows': 183, 'missing': 0, 'content_changed': 0}
- 上一轮 test.jsonl 的 sha256:c665a541f693e3803982c3cc3ba5f2991ae8af4003fe8b80cc3ba6d727c0e91a

## 2. 逐题清单

### eval_price_001 | intent = `price`

- **Question**:杨枝甘露和茉莉绿茶,哪个更贵?
- **Gold answer**:杨枝甘露 18 元一杯,比茉莉绿茶(8 元一杯)贵 10 元。
- **Keywords**:['18', '8']
- **Knowledge source**:kb.py PRODUCTS(杨枝甘露 18 / 茉莉绿茶 8)
- **Novelty reason**:跨产品比价,已有 price 只有单款问价与"最便宜的是什么"。
- **family_id**:`fam_5e18ea147f5b`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_002 | intent = `price`

- **Question**:菜单上最贵的是哪一款?
- **Gold answer**:最贵的是杨枝甘露,18 元一杯。
- **Keywords**:['杨枝甘露', '18']
- **Knowledge source**:kb.py PRODUCTS(8 款价格 8~18 元,最高为杨枝甘露 18)
- **Novelty reason**:全场最高价查询,与已有的"最便宜的是什么"方向相反。
- **family_id**:`fam_bbcd513a4c52`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_003 | intent = `price`

- **Question**:10 元以内能买到哪几款饮品?
- **Gold answer**:10 元以内的有茉莉绿茶 8 元和柠檬气泡水 10 元。
- **Keywords**:['茉莉绿茶', '柠檬气泡水']
- **Knowledge source**:kb.py PRODUCTS(茉莉绿茶 8、柠檬气泡水 10)
- **Novelty reason**:价格区间筛选,不是单款问价。
- **family_id**:`fam_c9567afb80f8`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_004 | intent = `price`

- **Question**:珍珠和椰果的加价是一样的吗?
- **Gold answer**:一样,珍珠和椰果都是加 2 元。
- **Keywords**:['2']
- **Knowledge source**:kb.py TOPPINGS(珍珠 2、椰果 2)
- **Novelty reason**:两种小料之间的比价,已有的是单种小料问价。
- **family_id**:`fam_c6960ccf6211`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_005 | intent = `price`

- **Question**:五种小料每种各加一份,一共要多花多少钱?
- **Gold answer**:珍珠 2 元、椰果 2 元、芋泥 3 元、奶盖 4 元、布丁 3 元,一共 14 元。
- **Keywords**:['14']
- **Knowledge source**:kb.py TOPPINGS(2+2+3+4+3=14)
- **Novelty reason**:全部小料的合计加价,是新场景(不是单种小料价格)。
- **family_id**:`fam_01adf0fa6ff2`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_006 | intent = `price`

- **Question**:经典珍珠奶茶加椰果和芋泥,一共多少钱?
- **Gold answer**:经典珍珠奶茶 12 元,加椰果 2 元、芋泥 3 元,一共 17 元。
- **Keywords**:['17']
- **Knowledge source**:kb.py PRODUCTS(经典珍珠奶茶 12)+ TOPPINGS(椰果 2、芋泥 3)
- **Novelty reason**:单品 + 多种小料的组合计价,KB 支持但此前没有这种问法。
- **family_id**:`fam_f782eae03af8`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_007 | intent = `price`

- **Question**:来两杯茉莉绿茶,一共多少钱?
- **Gold answer**:茉莉绿茶 8 元一杯,两杯一共 16 元。
- **Keywords**:['16']
- **Knowledge source**:kb.py PRODUCTS(茉莉绿茶 8,8×2=16)
- **Novelty reason**:多杯总价,已有问法都是单杯价格。
- **family_id**:`fam_7c0c218be346`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_008 | intent = `price`

- **Question**:周三买两杯经典珍珠奶茶,一共要付多少钱?
- **Gold answer**:经典珍珠奶茶 12 元一杯,周三第二杯半价,两杯一共 18 元。
- **Keywords**:['18']
- **Knowledge source**:kb.py PRODUCTS(12)+ COUPON_WED(第二杯半价),12+6=18
- **Novelty reason**:把会员日优惠换算成实付金额,是 price×member 的新场景。
- **family_id**:`fam_4f010654f024`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_009 | intent = `price`

- **Question**:新客第一次买一杯杨枝甘露,实际要付多少钱?
- **Gold answer**:杨枝甘露 18 元一杯,新客注册立减 3 元,实际 15 元。
- **Keywords**:['15']
- **Knowledge source**:kb.py PRODUCTS(18)+ COUPON_NEW(立减 3),18-3=15
- **Novelty reason**:新客优惠后的实际支付价,已有的是优惠本身而非算价。
- **family_id**:`fam_8d445ba76cc8`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_010 | intent = `price`

- **Question**:我有 100 积分,买一杯茉莉绿茶能抵多少?
- **Gold answer**:100 积分可抵 5 元,茉莉绿茶 8 元一杯,抵扣后是 3 元。
- **Keywords**:['5', '3']
- **Knowledge source**:kb.py MEMBER_RULE(100 积分抵 5 元)+ PRODUCTS(茉莉绿茶 8)
- **Novelty reason**:积分抵扣后的实付价,不重复已有的"积分有什么用"。
- **family_id**:`fam_34eed05de63c`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_011 | intent = `price`

- **Question**:想凑到免配送费,最少要买多少钱?
- **Gold answer**:满 25 元免配送费,所以最少要买到 25 元。
- **Keywords**:['25']
- **Knowledge source**:kb.py DELIVERY_FREE_OVER=25
- **Novelty reason**:凑单门槛金额,是 price×delivery 的新场景(不是问配送费多少)。
- **family_id**:`fam_e8cd403dded1`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_price_012 | intent = `price`

- **Question**:一杯芝士奶盖四季春加上 3 元配送费,一共多少钱?
- **Gold answer**:芝士奶盖四季春 16 元一杯,加 3 元配送费一共 19 元。
- **Keywords**:['19']
- **Knowledge source**:kb.py PRODUCTS(16)+ DELIVERY_FEE(3),16+3=19
- **Novelty reason**:商品加配送费的总额计算,已有价格题都不含配送费。
- **family_id**:`fam_e4db8d139ed6`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_001 | intent = `recommend`

- **Question**:想喝清淡一点的,推荐哪款?
- **Gold answer**:推荐茉莉绿茶,8 元一杯,标签就是清爽清淡。
- **Keywords**:['茉莉绿茶']
- **Knowledge source**:kb.py PRODUCTS 标签(茉莉绿茶 tags 含"清淡")
- **Novelty reason**:"清淡"不在已有模板使用的 16 个 tag 里,是新的筛选维度。
- **family_id**:`fam_c1f8262813de`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_002 | intent = `recommend`

- **Question**:有带西米的饮品吗?
- **Gold answer**:杨枝甘露里含西米,18 元一杯,喜欢颗粒口感可以试试。
- **Keywords**:['杨枝甘露']
- **Knowledge source**:kb.py PRODUCTS 标签(杨枝甘露 tags 含"西米")
- **Novelty reason**:按配料"西米"推荐,不在已有模板的 tag 集合中。
- **family_id**:`fam_f56e251c93ac`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_003 | intent = `recommend`

- **Question**:第一次来不知道怎么点,帮我搭一个两杯的入门组合?
- **Gold answer**:可以点一杯杨枝甘露 18 元配一杯经典珍珠奶茶 12 元,一杯果味一杯奶茶,一共 30 元。
- **Keywords**:['杨枝甘露', '经典珍珠奶茶']
- **Knowledge source**:kb.py PRODUCTS(两款均为 8 款在售产品之一);seeds.py seed_015 已确认这两款是招牌
- **Novelty reason**:首次购买的组合点单,已有都是单品推荐。
- **family_id**:`fam_97e6bf0a1a75`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_004 | intent = `recommend`

- **Question**:我乳糖不耐受,推荐一款不含牛奶的饮品?
- **Gold answer**:过敏原提示里不含牛奶、乳制品的饮品有茉莉绿茶 8 元和柠檬气泡水 10 元,这两款更稳妥。
- **Keywords**:['茉莉绿茶', '柠檬气泡水']
- **Knowledge source**:kb.py ALLERGENS(仅黑糖珍珠鲜奶/芋泥波波鲜奶含牛奶、芝士奶盖四季春含乳制品)
- **Novelty reason**:过敏约束下的推荐,区别于已有的"哪些饮品含牛奶"(纯查询)。
- **family_id**:`fam_b68fdad4f0f6`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_005 | intent = `recommend`

- **Question**:两个人喝,想一杯果味一杯奶茶,怎么点比较合适?
- **Gold answer**:可以点杨枝甘露 18 元(芒果、西柚风味)配经典珍珠奶茶 12 元,一共 30 元。
- **Keywords**:['杨枝甘露', '经典珍珠奶茶']
- **Knowledge source**:kb.py PRODUCTS 标签(杨枝甘露含芒果/西柚;经典珍珠奶茶含奶茶/珍珠)
- **Novelty reason**:双人双口味搭配,不是单款推荐。
- **family_id**:`fam_383ad9424bb2`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_006 | intent = `recommend`

- **Question**:我想一次喝到两种水果风味,推荐哪款?
- **Gold answer**:推荐杨枝甘露,含芒果和西柚两种水果风味,18 元一杯。
- **Keywords**:['杨枝甘露']
- **Knowledge source**:kb.py PRODUCTS 标签(杨枝甘露 tags: 芒果、西柚)
- **Novelty reason**:按"双水果风味"这一组合条件推荐,不同于单一 tag(芒果/西柚)的已有问法。
- **family_id**:`fam_a09cd7b165cb`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_007 | intent = `recommend`

- **Question**:想给同事带 4 杯不重样的饮品,有什么搭配建议?
- **Gold answer**:可以点经典珍珠奶茶 12 元、焦糖布丁奶茶 14 元、茉莉绿茶 8 元、杨枝甘露 18 元,口味覆盖奶茶、茶和果味,一共 52 元。
- **Keywords**:['52']
- **Knowledge source**:kb.py PRODUCTS(12+14+8+18=52)
- **Novelty reason**:多人多杯不重样搭配,是新的点单场景。
- **family_id**:`fam_9e92ec25bcde`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_008 | intent = `recommend`

- **Question**:你们家有哪些茶类饮品?
- **Gold answer**:茶类的有茉莉绿茶 8 元(标签含绿茶)和芝士奶盖四季春 16 元(标签含茶)。
- **Keywords**:['茉莉绿茶', '芝士奶盖四季春']
- **Knowledge source**:kb.py PRODUCTS 标签(茉莉绿茶含"绿茶"、芝士奶盖四季春含"茶")
- **Novelty reason**:"茶类"是跨产品的归纳,已有模板 tag 只有绿茶/茉莉/芝士/奶盖等具体词。
- **family_id**:`fam_acd3ac68783a`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_009 | intent = `recommend`

- **Question**:想要芋泥但不想喝鲜奶类,有别的办法吗?
- **Gold answer**:可以加芋泥小料,加 3 元,比如加在茉莉绿茶上就是一杯带芋泥的茶饮。
- **Keywords**:['芋泥', '3']
- **Knowledge source**:kb.py TOPPINGS(芋泥 3 元)+ PRODUCTS(茉莉绿茶 8)
- **Novelty reason**:用"小料"绕开产品限制的推荐思路,KB 支持但此前完全没有。
- **family_id**:`fam_bebd6d6669bd`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_010 | intent = `recommend`

- **Question**:一杯奶茶配一杯清爽的茶,怎么组合?
- **Gold answer**:可以经典珍珠奶茶 12 元配茉莉绿茶 8 元,一浓一淡,一共 20 元。
- **Keywords**:['经典珍珠奶茶', '茉莉绿茶']
- **Knowledge source**:kb.py PRODUCTS(12 与 8,合计 20)
- **Novelty reason**:浓淡搭配的组合推荐,区别于单款推荐和双人场景。
- **family_id**:`fam_343f5e51fecf`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_011 | intent = `recommend`

- **Question**:想加点椰果,推荐怎么搭配?
- **Gold answer**:椰果加 2 元,可以加在经典珍珠奶茶(12 元)上,一共 14 元。
- **Keywords**:['椰果']
- **Knowledge source**:kb.py TOPPINGS(椰果 2)+ PRODUCTS(经典珍珠奶茶 12)
- **Novelty reason**:小料搭配推荐,已有的是"加椰果多少钱"(纯问价)。
- **family_id**:`fam_42af259f76c2`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_recommend_012 | intent = `recommend`

- **Question**:四个人一人一杯,预算 60 元够吗?
- **Gold answer**:够。最便宜的四款是茉莉绿茶 8 元、柠檬气泡水 10 元、经典珍珠奶茶 12 元、焦糖布丁奶茶 14 元,一共 44 元,在 60 元预算内。
- **Keywords**:['44']
- **Knowledge source**:kb.py PRODUCTS(8+10+12+14=44,全部 ≤60)
- **Novelty reason**:按人数与预算做可行性判断,是新的决策场景。
- **family_id**:`fam_0967d7ed35bd`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_001 | intent = `after_sale`

- **Question**:饮品做错口味了,可以怎么处理?
- **Gold answer**:出品 30 分钟内口味不满意,可以免费重做或退款。
- **Keywords**:['30', '重做', '退款']
- **Knowledge source**:kb.py REFUND_RULE
- **Novelty reason**:做错单场景,已有是"不好喝/味道不对"这类主观口味问题。
- **family_id**:`fam_0c7d73b3c813`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_002 | intent = `after_sale`

- **Question**:冰量做错了,能重新做一杯吗?
- **Gold answer**:可以的,出品 30 分钟内口味不满意可以免费重做或退款,冰量做错可以直接申请重做。
- **Keywords**:['30', '重做']
- **Knowledge source**:kb.py REFUND_RULE
- **Novelty reason**:规格(冰量)出错,不是口味评价类问题。
- **family_id**:`fam_19add41290d1`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_003 | intent = `after_sale`

- **Question**:我想申请退款,需要满足什么条件?
- **Gold answer**:两种情形:饮品出品 30 分钟内口味不满意可免费重做或退款;外卖洒漏或破损,拍照联系客服可退款。
- **Keywords**:['30', '退款', '拍照']
- **Knowledge source**:kb.py REFUND_RULE + DELIVERY_REFUND
- **Novelty reason**:退款条件汇总,已有的是单纯问"能退款吗"。
- **family_id**:`fam_a643f68fced3`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_004 | intent = `after_sale`

- **Question**:外卖杯盖没盖好洒出来了,怎么退?
- **Gold answer**:外卖洒漏或破损,拍照联系客服可退款。
- **Keywords**:['拍照', '退款']
- **Knowledge source**:kb.py DELIVERY_REFUND
- **Novelty reason**:具体破损原因(杯盖)下的退款规定,已有只问"外卖洒了怎么办"。
- **family_id**:`fam_6e52e2e84aaa`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_005 | intent = `after_sale`

- **Question**:超过 30 分钟才觉得不好喝,还能退吗?
- **Gold answer**:免费重做或退款对应的是"出品 30 分钟内口味不满意",超过 30 分钟不在这个规则范围内。
- **Keywords**:['30']
- **Knowledge source**:kb.py REFUND_RULE(30 分钟时限边界)
- **Novelty reason**:政策时限的边界询问,已有问题都默认在时限内。
- **family_id**:`fam_92f0e5da946a`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_006 | intent = `after_sale`

- **Question**:我是自取的,也能免费重做吗?
- **Gold answer**:可以,规则是饮品出品 30 分钟内口味不满意可免费重做或退款。
- **Keywords**:['30', '重做']
- **Knowledge source**:kb.py REFUND_RULE(以"出品"为口径,不区分自取/堂食)
- **Novelty reason**:购买渠道(自取)场景,已有问题都没提渠道。
- **family_id**:`fam_3ad1684f62b1`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_007 | intent = `after_sale`

- **Question**:我可以既要重做又要退款吗?
- **Gold answer**:规则是二选一:出品 30 分钟内口味不满意,可以免费重做或退款。
- **Keywords**:['重做', '退款']
- **Knowledge source**:kb.py REFUND_RULE("免费重做或退款"为二选一)
- **Novelty reason**:权益边界(能否同时主张),是新的政策解读场景。
- **family_id**:`fam_ce4aba115f24`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_008 | intent = `after_sale`

- **Question**:外卖洒了,是找骑手还是找客服?
- **Gold answer**:外卖洒漏或破损,拍照联系客服可退款,走客服渠道处理。
- **Keywords**:['拍照', '客服']
- **Knowledge source**:kb.py DELIVERY_REFUND(联系客服)
- **Novelty reason**:责任/渠道询问,已有问题只问结果不问找谁。
- **family_id**:`fam_93878c0aea2c`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_009 | intent = `after_sale`

- **Question**:能退款的情况有哪些?
- **Gold answer**:两种:饮品出品 30 分钟内口味不满意可免费重做或退款;外卖洒漏或破损,拍照联系客服可退款。
- **Keywords**:['30', '退款']
- **Knowledge source**:kb.py REFUND_RULE + DELIVERY_REFUND
- **Novelty reason**:把两条售后政策合并枚举,已有问题都是单一场景。
- **family_id**:`fam_ee9f4ecd4b03`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_after_sale_010 | intent = `after_sale`

- **Question**:重做或退款的申请要在多久之内提出?
- **Gold answer**:要在出品 30 分钟内提出,30 分钟内口味不满意可以免费重做或退款。
- **Keywords**:['30']
- **Knowledge source**:kb.py REFUND_RULE
- **Novelty reason**:直接问时限本身,与已有"不好喝能换吗"这类结果导向问题不同。
- **family_id**:`fam_065b8c234427`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_001 | intent = `allergen`

- **Question**:芝士奶盖四季春含乳制品吗?
- **Gold answer**:含的,芝士奶盖四季春的过敏原提示是含乳制品。
- **Keywords**:['乳制品']
- **Knowledge source**:kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)
- **Novelty reason**:单款产品的乳制品确认,已有是"哪些饮品是奶制品"这类全表枚举。
- **family_id**:`fam_3d1c8fb69486`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_002 | intent = `allergen`

- **Question**:黑糖珍珠鲜奶的过敏原是什么?
- **Gold answer**:黑糖珍珠鲜奶含牛奶。
- **Keywords**:['牛奶']
- **Knowledge source**:kb.py ALLERGENS(黑糖珍珠鲜奶: 含牛奶)
- **Novelty reason**:按产品查过敏原条目,已有只查过杨枝甘露和"哪款不含奶"。
- **family_id**:`fam_b2e751bf7624`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_003 | intent = `allergen`

- **Question**:芋泥波波鲜奶里含哪两种过敏原?
- **Gold answer**:芋泥波波鲜奶含牛奶和芋泥。
- **Keywords**:['牛奶', '芋泥']
- **Knowledge source**:kb.py ALLERGENS(芋泥波波鲜奶: 含牛奶和芋泥)
- **Novelty reason**:双过敏原产品查询,是新的产品×过敏原组合。
- **family_id**:`fam_2e815d046a64`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_004 | intent = `allergen`

- **Question**:我对芒果过敏,能喝杨枝甘露吗?
- **Gold answer**:不建议,杨枝甘露含芒果,对芒果过敏的话不建议饮用。
- **Keywords**:['芒果']
- **Knowledge source**:kb.py ALLERGENS(杨枝甘露: 含芒果)
- **Novelty reason**:过敏人群的决策询问,已有是"杨枝甘露里有芒果吗"这种事实查询。
- **family_id**:`fam_8e19de86fc32`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_005 | intent = `allergen`

- **Question**:我乳糖不耐受,能喝芝士奶盖四季春吗?
- **Gold answer**:不建议,芝士奶盖四季春含乳制品,乳糖不耐受的话要避开。
- **Keywords**:['乳制品']
- **Knowledge source**:kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)
- **Novelty reason**:特定人群(乳糖不耐受)+ 单款的判断,是新的组合场景。
- **family_id**:`fam_b773551c3f0d`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_006 | intent = `allergen`

- **Question**:同时对牛奶和芒果过敏,能喝哪几款?
- **Gold answer**:要同时避开含牛奶的黑糖珍珠鲜奶、芋泥波波鲜奶和含乳制品的芝士奶盖四季春,以及含芒果的杨枝甘露;剩下的经典珍珠奶茶、焦糖布丁奶茶、茉莉绿茶、柠檬气泡水不在这些过敏原提示里。
- **Keywords**:['经典珍珠奶茶', '茉莉绿茶']
- **Knowledge source**:kb.py ALLERGENS(4 条产品过敏原提示)
- **Novelty reason**:双过敏原求交集,需要综合全部 4 条过敏原记录,信息需求最复杂。
- **family_id**:`fam_12c9a990e881`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_007 | intent = `allergen`

- **Question**:哪款饮品同时含牛奶和芋泥?
- **Gold answer**:芋泥波波鲜奶,它的过敏原提示是含牛奶和芋泥。
- **Keywords**:['芋泥波波鲜奶']
- **Knowledge source**:kb.py ALLERGENS(芋泥波波鲜奶: 含牛奶和芋泥)
- **Novelty reason**:反向检索(按过敏原组合找产品),与"查某产品含什么"方向相反。
- **family_id**:`fam_ed84442390ad`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_008 | intent = `allergen`

- **Question**:奶盖类的饮品含乳制品吗?
- **Gold answer**:含的,芝士奶盖四季春含乳制品,奶盖类要注意乳制品过敏。
- **Keywords**:['乳制品']
- **Knowledge source**:kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)
- **Novelty reason**:按产品标签类别(奶盖)推断过敏原,是新的归纳式问法。
- **family_id**:`fam_e34b71ef553c`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_009 | intent = `allergen`

- **Question**:我牛奶过敏,黑糖珍珠鲜奶和芋泥波波鲜奶都不能喝吗?
- **Gold answer**:是的,黑糖珍珠鲜奶和芋泥波波鲜奶都含牛奶,牛奶过敏的话两款都要避开。
- **Keywords**:['黑糖珍珠鲜奶', '芋泥波波鲜奶']
- **Knowledge source**:kb.py ALLERGENS(两款均含牛奶)
- **Novelty reason**:多产品同时确认,已有只问过"有含牛奶的饮品吗"这类存在性问题。
- **family_id**:`fam_95a1a7ee3b38`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_allergen_010 | intent = `allergen`

- **Question**:对乳制品过敏的人,菜单上要避开哪一款?
- **Gold answer**:要避开芝士奶盖四季春,它的过敏原提示是含乳制品。
- **Keywords**:['芝士奶盖四季春']
- **Knowledge source**:kb.py ALLERGENS(芝士奶盖四季春: 含乳制品)
- **Novelty reason**:从过敏人群出发做单品定位,与已有"哪些是奶制品"的枚举方向不同。
- **family_id**:`fam_177f0bad1fc4`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_001 | intent = `hours`

- **Question**:你们一天营业多少个小时?
- **Gold answer**:营业时间是 10:00-22:00,一天共 12 个小时。
- **Keywords**:['12']
- **Knowledge source**:kb.py OPEN_TIME/CLOSE_TIME(10:00-22:00 = 12 小时)
- **Novelty reason**:营业时长换算,已有只问开门/关门时间点。
- **family_id**:`fam_e6b5cf2f7003`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_002 | intent = `hours`

- **Question**:节假日你们营业吗?
- **Gold answer**:营业,节假日正常营业,营业时间 10:00-22:00。
- **Keywords**:['10:00', '22:00']
- **Knowledge source**:kb.py OPEN_TIME/CLOSE_TIME;seeds.py seed_028 确认"周末和节假日正常营业"
- **Novelty reason**:节假日场景,已有只问过周末。
- **family_id**:`fam_45e78ac1813f`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_003 | intent = `hours`

- **Question**:早上九点能买到吗?
- **Gold answer**:买不到,门店 10:00 才开门营业。
- **Keywords**:['10:00']
- **Knowledge source**:kb.py OPEN_TIME=10:00
- **Novelty reason**:以"某个具体时间点能不能买到"提问,是新的判定场景。
- **family_id**:`fam_5c27c10fab8d`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_004 | intent = `hours`

- **Question**:晚上十点半还能到店买吗?
- **Gold answer**:不能,门店 22:00 关门,十点半已经打烊了。
- **Keywords**:['22:00']
- **Knowledge source**:kb.py CLOSE_TIME=22:00
- **Novelty reason**:打烊后的边界时间判断,与已有"营业到几点"不同。
- **family_id**:`fam_58d0b02ed064`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_005 | intent = `hours`

- **Question**:每天的营业时间都一样吗?
- **Gold answer**:一样,都是 10:00-22:00。
- **Keywords**:['10:00', '22:00']
- **Knowledge source**:kb.py OPEN_TIME/CLOSE_TIME;seeds.py seed_028 确认周末/节假日同时间
- **Novelty reason**:作息一致性确认,是新的政策澄清类问题。
- **family_id**:`fam_6bbb1a23a7ed`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_006 | intent = `hours`

- **Question**:外卖的结束时间和门店关门时间一样吗?
- **Gold answer**:不一样,门店 22:00 关门,外卖 21:30 截止下单。
- **Keywords**:['22:00', '21:30']
- **Knowledge source**:kb.py CLOSE_TIME=22:00 与 DELIVERY_CUTOFF=21:30
- **Novelty reason**:两个时间的对比,需要同时检索门店与外送政策。
- **family_id**:`fam_cfb463104069`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_007 | intent = `hours`

- **Question**:中午去店里,你们在营业吗?
- **Gold answer**:在的,营业时间是 10:00-22:00,中午也在营业时间内。
- **Keywords**:['10:00', '22:00']
- **Knowledge source**:kb.py OPEN_TIME/CLOSE_TIME
- **Novelty reason**:特定时段(中午)是否营业,已有问的是"现在这个点"。
- **family_id**:`fam_3d0624af8ede`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_hours_008 | intent = `hours`

- **Question**:最晚几点还能进店点单?
- **Gold answer**:门店营业到 22:00,最晚在 22:00 关门前到店点单。
- **Keywords**:['22:00']
- **Knowledge source**:kb.py CLOSE_TIME=22:00
- **Novelty reason**:以"最晚点单时间"提问,是新的可用性判断角度。
- **family_id**:`fam_081b5cb8a968`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_001 | intent = `delivery`

- **Question**:3.5 公里能送吗?
- **Gold answer**:不能,门店只送 3 公里以内。
- **Keywords**:['3']
- **Knowledge source**:kb.py DELIVERY_RADIUS=3 公里
- **Novelty reason**:具体距离的可行性判断,已有只问"配送范围多远"。
- **family_id**:`fam_55b7957f03e8`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_002 | intent = `delivery`

- **Question**:买满多少钱可以免配送费?
- **Gold answer**:满 25 元免配送费。
- **Keywords**:['25']
- **Knowledge source**:kb.py DELIVERY_FREE_OVER=25
- **Novelty reason**:免配送门槛,已有问的是"配送费多少钱"。
- **family_id**:`fam_f21d61dc0afe`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_003 | intent = `delivery`

- **Question**:只点一杯茉莉绿茶,要付配送费吗?
- **Gold answer**:要付,茉莉绿茶 8 元不满 25 元,配送费 3 元。
- **Keywords**:['3', '25']
- **Knowledge source**:kb.py PRODUCTS(茉莉绿茶 8)与 DELIVERY_FEE=3/DELIVERY_FREE_OVER=25
- **Novelty reason**:单品金额与免配送门槛的对照判断,是新的组合计算。
- **family_id**:`fam_1f27547c7604`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_004 | intent = `delivery`

- **Question**:点两杯杨枝甘露够免配送费吗?
- **Gold answer**:够,两杯杨枝甘露 36 元,超过 25 元可以免配送费。
- **Keywords**:['36', '25']
- **Knowledge source**:kb.py PRODUCTS(杨枝甘露 18×2=36)与 DELIVERY_FREE_OVER=25
- **Novelty reason**:用具体订单金额判断能否免配送费。
- **family_id**:`fam_9d825e267091`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_005 | intent = `delivery`

- **Question**:晚上九点四十还能点外卖吗?
- **Gold answer**:不能,外卖 21:30 截止下单,九点四十已经过了截止时间。
- **Keywords**:['21:30']
- **Knowledge source**:kb.py DELIVERY_CUTOFF=21:30
- **Novelty reason**:具体时间点的下单可行性判断,已有只问截止时间。
- **family_id**:`fam_4322f9e4646f`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_delivery_006 | intent = `delivery`

- **Question**:配送范围是以门店为中心算的吗?
- **Gold answer**:是的,门店 3 公里内配送,以门店为中心。
- **Keywords**:['3']
- **Knowledge source**:kb.py DELIVERY_RADIUS=3 公里
- **Novelty reason**:配送范围的计量口径(以门店为中心),是新的政策澄清问题。
- **family_id**:`fam_32749a1cd215`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_member_001 | intent = `member`

- **Question**:消费 30 元能积多少分?
- **Gold answer**:消费 1 元积 1 分,消费 30 元可以积 30 分。
- **Keywords**:['30']
- **Knowledge source**:kb.py MEMBER_RULE(1 元 1 分)
- **Novelty reason**:具体消费金额换算积分,已有只问"怎么积分"。
- **family_id**:`fam_1c5a898059f5`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_member_002 | intent = `member`

- **Question**:500 积分能抵多少钱?
- **Gold answer**:100 积分抵 5 元,500 积分可以抵 25 元。
- **Keywords**:['25']
- **Knowledge source**:kb.py MEMBER_RULE(100 积分抵 5 元,500 分 = 25 元)
- **Novelty reason**:具体积分数的抵扣换算,已有只问"积分有什么用"。
- **family_id**:`fam_964ce1dac81e`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_member_003 | intent = `member`

- **Question**:消费满 100 元能积多少分?
- **Gold answer**:消费 1 元积 1 分,100 元可以积 100 分。
- **Keywords**:['100']
- **Knowledge source**:kb.py MEMBER_RULE
- **Novelty reason**:换一个消费档位做积分换算,与 30 元档位是不同计算实例。
- **family_id**:`fam_1d254556c496`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_member_004 | intent = `member`

- **Question**:消费一元能积几分?
- **Gold answer**:消费 1 元积 1 分。
- **Keywords**:['1']
- **Knowledge source**:kb.py MEMBER_RULE
- **Novelty reason**:直接问积分单位比例,是把既有规则拆到最小颗粒度来考。
- **family_id**:`fam_0b872956069d`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_member_005 | intent = `member`

- **Question**:我是第一次来,有什么优惠吗?
- **Gold answer**:有,新客注册立减 3 元。
- **Keywords**:['3']
- **Knowledge source**:kb.py COUPON_NEW(新客注册立减 3 元)
- **Novelty reason**:首单新客视角,已有问的是"有优惠券吗/怎么成为会员"。
- **family_id**:`fam_663bbd6391f9`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_store_001 | intent = `store`

- **Question**:茶语时光主打什么?
- **Gold answer**:茶语时光 2019 年创立,主打新鲜现做的茶饮。
- **Keywords**:['新鲜现做']
- **Knowledge source**:kb.py BRAND_STORY
- **Novelty reason**:品牌定位询问,已有的是"你们是什么品牌"这种身份问题。
- **family_id**:`fam_1168e05ef30f`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_store_002 | intent = `store`

- **Question**:你们官方小程序能做什么?
- **Gold answer**:可以在小程序里查看离你最近的门店。
- **Keywords**:['小程序']
- **Knowledge source**:kb.py STORE_NOTE(可通过小程序查看最近门店)
- **Novelty reason**:问小程序的功能,已有的是"怎么找最近门店"这一步操作。
- **family_id**:`fam_5b9d78108e59`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_store_003 | intent = `store`

- **Question**:你们有官方小程序吗?
- **Gold answer**:有的,可以通过茶语时光小程序查看最近的门店。
- **Keywords**:['小程序']
- **Knowledge source**:kb.py STORE_NOTE
- **Novelty reason**:存在性确认(有没有小程序),与"怎么用小程序找店"不同。
- **family_id**:`fam_5e102995d02d`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_store_004 | intent = `store`

- **Question**:你们的门店多吗?
- **Gold answer**:门店不少,全国有 200 多家。
- **Keywords**:['200']
- **Knowledge source**:kb.py STORE_NOTE(全国 200+ 家门店)
- **Novelty reason**:规模化描述询问,不同于"附近有门店吗"这种定位问题。
- **family_id**:`fam_cdd4bfca2b89`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_store_005 | intent = `store`

- **Question**:门店提供哪些额外服务?
- **Gold answer**:门店提供免费 WiFi。
- **Keywords**:['WiFi']
- **Knowledge source**:kb.py WIFI_NOTE
- **Novelty reason**:服务清单询问,已有的是"有 WiFi 吗"这种单点确认。
- **family_id**:`fam_2990d26e0f45`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_001 | intent = `ood` | OOD

- **Question**:帮我写一段 Python 爬虫代码
- **Gold answer**:抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。
- **Keywords**:['抱歉', '无法回答']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES(知识库外统一拒答)
- **Novelty reason**:编程类 OOD,现有 OOD 主题里没有编程。
- **family_id**:`fam_58ef06f847ee`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_002 | intent = `ood` | OOD

- **Question**:帮我算一下 128 乘以 37 等于多少
- **Gold answer**:不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。
- **Keywords**:['不好意思', '服务范围']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:数学计算类 OOD,现有 OOD 没有纯计算题。
- **family_id**:`fam_742fd455d1f8`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_003 | intent = `ood` | OOD

- **Question**:帮我规划一下去三亚的五天行程
- **Gold answer**:抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。
- **Keywords**:['抱歉', '无法回答']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:旅游规划类 OOD(已有只有"退机票/订酒店"这类票务操作)。
- **family_id**:`fam_7f9dc2e9b40e`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_004 | intent = `ood` | OOD

- **Question**:隔壁那家奶茶店有什么新品?
- **Gold answer**:不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。
- **Keywords**:['不好意思', '服务范围']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:其他品牌类 OOD,测试模型是否会为竞品编造信息(现有 OOD 没有竞品问题)。
- **family_id**:`fam_21e26627b168`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_005 | intent = `ood` | OOD

- **Question**:我租房合同有纠纷,能帮我看看吗?
- **Gold answer**:抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。
- **Keywords**:['抱歉', '无法回答']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:法律咨询类 OOD(已有"附近哪里可以租房"是生活服务类,主题不同)。
- **family_id**:`fam_66ad4c0a1c76`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_006 | intent = `ood` | OOD

- **Question**:我家猫最近不吃东西怎么办?
- **Gold answer**:不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。
- **Keywords**:['不好意思', '服务范围']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:宠物咨询类 OOD,现有 OOD 主题完全没有宠物。
- **family_id**:`fam_ba051a1e418a`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_007 | intent = `ood` | OOD

- **Question**:昨晚那场球赛谁赢了?
- **Gold answer**:抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。
- **Keywords**:['抱歉', '无法回答']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:体育赛事类 OOD,与已有的生活服务主题不同。
- **family_id**:`fam_57878a29e1c4`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_008 | intent = `ood` | OOD

- **Question**:推荐几部好看的电影
- **Gold answer**:不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。
- **Keywords**:['不好意思', '服务范围']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:影视推荐类 OOD,测试模型是否会把"推荐"能力错误地用到领域外。
- **family_id**:`fam_26eb7fd9d38d`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_009 | intent = `ood` | OOD

- **Question**:帮我把这句话翻译成英文
- **Gold answer**:抱歉,我是茶语时光的客服,只负责门店产品、营业时间、会员优惠和售后相关问题,这个问题我无法回答。
- **Keywords**:['抱歉', '无法回答']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:翻译类 OOD,属于通用语言能力请求,不在客服职责内。
- **family_id**:`fam_662f009ee783`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

### eval_ood_010 | intent = `ood` | OOD

- **Question**:番茄炒蛋怎么做才好吃?
- **Gold answer**:不好意思,这个问题超出了我的服务范围,建议您咨询相关专业渠道。我是茶语时光客服,只解答店内业务。
- **Keywords**:['不好意思', '服务范围']
- **Knowledge source**:kb.py REFUSAL_TEMPLATES
- **Novelty reason**:菜谱类 OOD,与饮品制作不同,属于领域外生活问题。
- **family_id**:`fam_d2a5f9c68336`
- **Leakage check**:PASS(exact / base / family 与 train、val、旧 test 均无交集)

## 3. 知识库无法支持、因此刻意没有造题的场景

| intent | 场景 | 缺失的事实 |
|---|---|---|
| price | 杯型/规格价格(中杯/大杯/升杯加价) | kb.py PRODUCTS 只有单一价格,没有杯型/容量字段 |
| price | 套餐价格、季节限定价格 | kb.py 没有套餐与季节限定商品 |
| allergen | 坚果/花生/芝麻过敏原,小料的过敏原 | kb.py ALLERGENS 只覆盖 4 款产品的 4 类提示,没有小料与其他过敏原 |
| after_sale | 漏发/错发/少送、异物、超时未送达的处理规则 | kb.py 只有 REFUND_RULE 与 DELIVERY_REFUND 两条政策,未规定漏发与超时 |
| member | 积分有效期、积分与优惠券叠加、生日福利 | kb.py MEMBER_RULE / COUPON_* 未涉及有效期与叠加规则 |
| store | 加盟合作、WiFi 密码、座位/环境、门店地址清单 | kb.py STORE_NOTE 只有门店数量与小程序查店,没有上述信息 |
| delivery | 超出 3 公里的处理、配送时长承诺、配送费退还 | kb.py 只定义 3 公里/3 元/满 25 元免,未定义超范围与时效承诺 |
| customize | 常温/加冰量以外的规格定制(如加浓、换奶) | kb.py 未定义糖度/冰量以外的定制选项 |

## 4. Assertions

| assertion | result | detail |
|---|---|---|
| gold_eval samples all present and flagged eval_only | PASS | expected=78 actual=78 |
| gold_eval only in test (train/val gold_eval = 0) | PASS | test=78 train=0 val=0 |
| gold_eval zero exact/base/family overlap with train/val/old_test | PASS | {'train': {'exact': 0, 'base': 0, 'family': 0}, 'val': {'exact': 0, 'base': 0, 'family': 0}, 'old_test': {'exact': 0, 'base': 0, 'family': 0}} |
| gold_eval OOD families disjoint from train/val/old_test OOD families | PASS | {'train': 0, 'val': 0, 'old_test': 0} |
| test covers every intent present in the dataset | PASS | test_intents=['after_sale', 'allergen', 'customize', 'delivery', 'hours', 'member', 'ood', 'price', 'recommend', 'store'] missing=[] |
| test has >=10 questions for price/recommend/after_sale/allergen | PASS | {'price': 12, 'recommend': 12, 'after_sale': 10, 'allergen': 10} |
| previous clean test rows preserved verbatim | PASS | {'previous_test_rows': 183, 'missing': 0, 'content_changed': 0} |
