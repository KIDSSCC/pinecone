import requests
import os
import urllib

class GetImage():
    def __init__(self,keyword='大雁',paginator=1):
        self.url = 'http://image.baidu.com/search/acjson?'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'}
        self.headers_image = {
            'User-Agent': 'Mozilla/5.0 (Windows\
                     NT 10.0; WOW64) AppleWebKit/537.36 \
                     (KHTML, like Gecko) Chrome/69.0.\
                    3497.81 Safari/537.36',
            'Referer': 'http://image.baidu.com/\
                    search/index?tn=baiduimage&ipn=r&\
                    ct=201326592&cl=2&lm=-1&st=-1&\
                    fm=result&fr=&sf=1&fmq=1557124645631_R&\
                    pv=&ic=&nc=1&z=&hd=1&latest=0&copyright\
                    =0&se=1&showtab=0&fb=0&width=&height=\
                    &face=0&istype=2&ie=utf-8&sid=&word=%\
                    E8%83%A1%E6%AD%8C'}
        # 定义关键词
        self.keyword = keyword
        # 定义要爬取的页数
        self.paginator = paginator

    def get_param(self):
        # 关键词转换为编码
        keyword = urllib.parse.quote(self.keyword)
        params = []
        for i in range(1, self.paginator + 1):
            params.append(
                'tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=1&latest=0&copyright=0&word={}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=star&pn={}&rn=30&gsm=78&1557125391211='.format(keyword, keyword, 30 * i))
        # 返回链接参数
        return params

    def get_urls(self, params):
        urls = []
        for param in params:
            # 拼接每页的链接
            urls.append(self.url + param)
        return urls  # 返回每页链接

    def get_image_url(self, urls):
        image_url = []
        for url in urls:
            json_data = requests.get(url, headers=self.headers).json()
            json_data = json_data.get('data')
            for i in json_data:
                if i:
                    image_url.append(i.get('thumbURL'))
        return image_url

    def get_image(self, image_url):
        """
        根据图片url，在本地目录下新建一个以搜索关键字命名的文件夹，然后将每一个图片存入。
        :param image_url:
        :return:
        """
        cwd = os.getcwd()
        file_name = os.path.join(cwd, self.keyword)
        if not os.path.exists(self.keyword):
            os.mkdir(file_name)
        for index, url in enumerate(image_url, start=1):
            with open(file_name + '/{}_0.jpg'.format(index), 'wb') as f:
                f.write(requests.get(url, headers=self.headers_image).content)
            if index != 0 and index % 30 == 0:
                print('第{}页下载完成'.format(index / 30))

    def __call__(self, *args, **kwargs):
        # 获取链接参数,param中包含了每页的一个链接
        params = self.get_param()
        # 根据参数获取url链接
        urls = self.get_urls(params)
        image_url = self.get_image_url(urls)
        # 返回了每一张图片的url
        self.get_image(image_url)

if __name__ == '__main__':
    spider = GetImage('星穹铁道', 3)
    spider()