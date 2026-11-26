"""Style samples for the "create" tailor mode.

``resume_senior_software`` stays as a plain module-level string (other code may
still reference it directly). Every sample exposed to the app is registered in
``STYLE_SAMPLES`` below as a ``StyleSample``.

To add a new sample:
1. Write the resume text as a triple-quoted string, following the same layout
   as ``resume_senior_software`` (name/contact line, then "个人简介"/"Summary",
   "工作经历"/"Experience", "技能"/"Skills", "教育背景"/"Education" sections).
   Experience entries are "Title, Company, Location" followed by a date line,
   then "- " bullets; continuation lines must be indented and start with no
   dash. Skills lines use "Label: a, b" or "Label：a、b" to render as a group.
2. Add a ``StyleSample(id=..., label=..., language=..., text=...)`` entry to
   ``STYLE_SAMPLES``. ``id`` must be unique and match ``^[A-Za-z0-9_-]+$``.
3. ``app/services/style_profile.py::compute_style_profile`` must be able to
   parse it — ``tests/unit/test_style_samples.py`` enforces this for every
   registered sample.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleSample:
    """A style sample the user can pick in the "create" tailor mode."""

    id: str
    label: str
    # ISO-ish content language code ("zh", "en", ...), matches get_language_name.
    language: str
    text: str


resume_senior_software = """
DANIEL OKAFOR
加拿大多伦多（远程） · daniel.okafor@example.com
github.com/dokafor-dev · linkedin.com/in/dokafor-dev

个人简介
高级全栈工程师，8 年经验，在 GCP 上用 TypeScript、React 和 Node.js
交付 B2B SaaS 产品。在 Brightmoor Commerce 带领 6 人团队完成单体到
服务的拆分，负责 2,400 个商户账户使用的结账与计费模块。

工作经历
高级软件工程师，Brightmoor Commerce，多伦多
2022 年 3 月至今
- 带领 6 名工程师，用 14 个月把 40 万行的 Rails 和 React 单体拆成
  5 个 TypeScript 服务，没有任何面向客户的停机；部署频率从每周一次
  提升到每天 20 次以上。
- 设计计费服务（Node.js、PostgreSQL、Stripe），每年处理 3800 万美元
  的商户交易额；通过调整催缴计划，把支付失败的重试减少 31%。
- 用 Next.js 的路由级代码拆分和服务端渲染，把商户仪表板的 Largest
  Contentful Paint 从 4.1 s 降到 1.6 s。
- 面试了 40 多位候选人，编写的前端招聘评分标准现已被 3 个团队采用。

软件工程师，Northgate Studio，温哥华
2020 年 8 月至 2022 年 2 月
- 为一款月活 9 万的设计工具构建实时协作层（WebSockets、CRDTs）。
- 用原生 API 和支持 tree-shaking 的导入替换 Moment 和 Lodash，前端
  包体积减少 45%。
- 为流量最高的 12 条流程引入 Playwright 端到端测试；UI 回归引发的
  事故从每季度约 5 起降到 1 起。

初级开发工程师，Ferrule Labs，温哥华
2018 年 6 月至 2020 年 7 月
- 在 Django 和 jQuery 代码库上为一款有 300 家客户的车队追踪产品
  交付功能。

技能
编程语言：TypeScript、JavaScript、Python、SQL
前端：React、Next.js、Playwright、Vite
后端：Node.js、Express、PostgreSQL、Redis、Stripe API
基础设施：GCP（Cloud Run、Cloud SQL、Pub/Sub）、Docker、Terraform、
GitHub Actions

教育背景
计算机科学理学学士（B.Sc.），Pacific Northern University，温哥华，2018
"""


STYLE_SAMPLES: tuple[StyleSample, ...] = (
    StyleSample(
        id="senior_fullstack_zh",
        label="高级全栈工程师（中文）",
        language="zh",
        text=resume_senior_software,
    ),
)

_STYLE_SAMPLES_BY_ID: dict[str, StyleSample] = {s.id: s for s in STYLE_SAMPLES}


def get_style_sample(sample_id: str) -> StyleSample | None:
    """Look up a registered style sample by id, or None when unknown."""
    return _STYLE_SAMPLES_BY_ID.get(sample_id)
