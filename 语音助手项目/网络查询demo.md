考虑到我们的助手功能可能比较单薄，而且双脑结构并不能增强它对指定问题的解答能力，所以我打算加入网络查询功能  
调研了一下，还是百度这种old厂厚道啊，每天还有100次免费调用次数 ~~百度还是个忠厚人呐~~  
[舰队航标](https://cloud.baidu.com/doc/qianfan-api/s/Hmbu8m06u)  
我们可以使用经典old搜索反复，也可以使用新推出的智能搜索。  
这个old搜索也就是百度搜索，它会返回数个网页中的内容，包括网址、内容、图片等等，需要自己提炼，如果对模型能力有信心的话可以用这种原始数据  
但是我们这里使用new！智能搜索，1次调用可以返回20条内容，包括网页原内容（和old差不多）+百度模型提炼内容，我们可以直接把提炼后的内容丢个ai，这样它可以更快捷地生成回答  

下面给一个最小示例代码，可以先去百度千帆平台注册一个账号，然后在刚才的舰队航标处阅读一下api调用文档，我们这里只是一个例子，没有使用全部功能：
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百度千帆智能搜索 Demo
使用智能搜索查询固定问题并输出返回内容
API Key 直接鉴权方式（每日100次免费额度）
文档：https://cloud.baidu.com/doc/qianfan-api/s/Hmbu8m06u
"""

import os
import requests
import json

# ============ 配置区域 ============
# 请在这里填写你的百度千帆 API Key
# 获取方式：https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application
QIANFAN_API_KEY = ""  # 只需要填 API Key

# 查询问题
QUERY = "2026年人工智能发展趋势"


def search_with_qianfan(query, api_key):
    """
    使用百度千帆智能搜索 API 进行搜索
    API Key 直接鉴权方式
    """
    # 智能搜索生成 API 端点
    url = "https://qianfan.baidubce.com/v2/ai_search"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"搜索请求失败: {response.status_code}, {response.text}")


def main():
    # 检查是否配置了 API Key
    api_key = QIANFAN_API_KEY or os.environ.get("QIANFAN_API_KEY", "")
    
    if not api_key:
        print("错误：请先在代码中配置 QIANFAN_API_KEY")
        print("获取方式：https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application")
        print("\n操作步骤：")
        print("1. 登录百度智能云控制台")
        print("2. 进入千帆大模型平台 → 应用接入")
        print("3. 创建应用，获取 API Key")
        print("4. 在应用详情页开通【智能搜索】服务权限")
        return
    
    print(f"正在查询: {QUERY}")
    print("-" * 50)
    
    try:
        # 执行搜索
        result = search_with_qianfan(QUERY, api_key)
        
        # 输出返回内容
        print("\n【API 返回内容】\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 提取并显示主要回答内容
        if "result" in result:
            print("\n" + "=" * 50)
            print("【智能搜索回答】\n")
            print(result["result"])
        
        # 显示搜索引用信息
        if "search_results" in result:
            print("\n" + "=" * 50)
            print("【搜索引用】\n")
            for idx, item in enumerate(result["search_results"], 1):
                print(f"[{idx}] {item.get('title', 'N/A')}")
                print(f"    {item.get('url', 'N/A')}\n")
                
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

```
