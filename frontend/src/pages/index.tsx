import React, { FC, useRef } from 'react';
import { Layout, Divider, Tabs, Button, Drawer, Switch, Tooltip } from 'antd'

const { TabPane } = Tabs
import { HistoryOutlined, QuestionCircleOutlined } from '@ant-design/icons';
const { Header, Sider, Content } = Layout
import PanelLeft from '@/components/panel-left';
import PanelRight from '@/components/panel-right';
import { useLocation, useHistory } from 'react-router-dom'; // 导入useHistory(v5)
import { useImmer } from 'use-immer';
import useModel from 'flooks';
import ProjectModel from '@/models/projectModel';
import History from './history';
import i18n from 'i18next';
import { help_doc_url } from '@/utils/constant';
import { useTranslation } from 'react-i18next';
import './index.less'

export interface IIndexProps { }
const Index: FC<IIndexProps> = (props) => {
  const { t } = useTranslation();
  const location = useLocation();
  const history = useHistory(); // 使用useHistory(v5)
  const historyRef = useRef()
  const [state, setState] = useImmer({
    showHistory: false,
  });
  const { setProject, curMode } = useModel(ProjectModel);
  
  // 根据路由设置当前模式
  React.useEffect(() => {
    let mode = 'guide'; // 默认模式
    
    if (location.pathname.includes('optimal')) {
      mode = 'optimal';
    } else if (location.pathname.includes('agent')) {
      mode = 'agent';
    } else if (location.pathname.includes('benchmark')) {
      mode = 'benchmark';
    } else if (location.pathname.includes('guide')) {
      mode = 'guide';
    }
    
    setProject({
      curMode: mode,
      result: null,
    });
  }, [location.pathname, setProject]);

  // 使用history.push实现跳转(v5)
  const onChangeMode = (mode: string) => {
    // 定义模式对应的路由
    const modeMap = {
      guide: '/guide',
      optimal: '/optimal',
      agent: '/agent',
      // benchmark: '/benchmark'
    };
    
    // 跳转路由
    history.push(modeMap[mode as keyof typeof modeMap] || '/guide', );

    window.location.reload()
    
    // 更新项目状态
    // setProject({
    //   curMode: mode,
    //   result: null,
    // });
  };
  
  const handleLanChange = (checked: boolean) => {
    i18n.changeLanguage(checked ? 'cn' : 'en')
  }
  
  return (
    <React.Fragment>
      <Layout className="llm-layout-wrapper" key={location.pathname}>
        <Header>
          <div className="header-wrapper">
            <div className={`${i18n.language === 'cn' ? 'header-logo1' : 'header-logo'}`}>
              <div></div>
            </div>
            <Divider type="vertical" />
            <div className="header-tabs">
              <Tabs activeKey={curMode} onChange={onChangeMode} animated={false}>
                <TabPane tab={t('guide mode')} key="guide" />
                <TabPane tab={t('optimal mode')} key="optimal" />
                <TabPane tab={t('agent mode')} key="agent" />
              </Tabs>
            </div>
            <div className="header-history">
              <Button type="primary" ghost icon={<HistoryOutlined />}
                onClick={() => { setState({ showHistory: true }) }}>
                {t('comparision')}
              </Button>
            </div>
            <div className="header-help">
              <Tooltip title={t('doc')}>
                <QuestionCircleOutlined onClick={() => { window.open('/calculator/#/help') }} />
              </Tooltip>
            </div>
            <div className="header-language">
              <Switch checkedChildren="中文" unCheckedChildren="English" onChange={handleLanChange}></Switch>
            </div>
          </div>
        </Header>
        <Layout className="llm-inner-layout-wrapper">
          <Sider  width={curMode === 'guide' || curMode === 'optimal' ? 430 : 400} theme='light'>
            <PanelLeft key={`left-${location.pathname}`}></PanelLeft>
          </Sider>
          <Content>
            <PanelRight key={`right-${location.pathname}`}></PanelRight>
          </Content>
        </Layout>
      </Layout>
      <Drawer title={t('comparision')} placement="right" width={900}
        onClose={() => { (historyRef?.current as any)?.handleClose() }}
        open={state.showHistory}>
        <History onClose={() => { setState({ showHistory: false }) }} ref={historyRef} />
      </Drawer>
    </React.Fragment>
  );
};

export default Index;
