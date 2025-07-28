import { FC, useEffect } from 'react';
import { useImmer } from 'use-immer';
import { Select, Divider, Input, InputNumber, Slider, Button, Drawer, message } from 'antd'
import Empty from '../empty';
import useModel from 'flooks';
import { getGpuList, getNetWork } from '@/services'
import ProjectModel from '@/models/projectModel';
import styles from './index.less';
import LogModel from '@/models/logModel';
import { PlusOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next';

const PARAMS_LIST = [
  {
    title: 'GPU Type',
    key: 'name'
  },
  {
    title: 'Sparse Tensor FP16 Processing power(Tflops)',
    key: 'sparse_tensor_fp16_processing_power'
  }, {
    title: 'FP32 Processing power(Tflops)',
    key: 'sparse_tensor_fp32_processing_power'
  },
  {
    title: 'Memory(GB)',
    key: 'memory'
  },
  {
    title: 'Memory Bandwidth(GB/s)',
    key: 'memory_bandwidth'
  },
  {
    title: 'Bus Bandwidth(GB/s)',
    key: 'bus_bandwidth'
  }, {
    title: 'support p2p',
    key: 'support_p2p'
  }
]
const SLIDER_LIST = [{
  title: 'Inter-host network bandwidth(Gb/s)',
  key: 'network_bandwidth',
  min: 0,
  max: 1600,
  precision: 1,
  step: 1
}]
export interface IGPUSelectionProps { }
const GpuSelection: FC<IGPUSelectionProps> = (props) => {
  const { setProject, curGpu, curNetwork } = useModel(ProjectModel);
  const { setChangeLog } = useModel(LogModel);
  const { t } = useTranslation();

  const handleItemClick = (key: string, item: any) => {
    setChangeLog('GPU', item?.name, curGpu?.name)
    setProject({
      curGpu: {
        ...item,
        num_procs: curGpu?.num_procs
      }
    });
  };


  const [state, setState] = useImmer({
    GPU_LIST: [] as any[],
    TOPO_LIST: [] as any[],
    showAddModal: false,
    newGpu: {} as any,
    netInfo: {} as any
  });

  const initCurGpu = () => {
    setProject({
      curGpu: {
        ...curGpu,
        num_procs: curGpu?.num_procs ? curGpu?.num_procs : 1
      }
    });
  }

  const loadGpuList = async () => {
    const localItems = JSON.parse(localStorage.getItem('local_gpus') || '[]') || []
    const gpuRes: any = await getGpuList()
    const gpuList = gpuRes.map((item: any) => {
      return {
        key: item.name,
        label: item.name,
        value: item.name,
        ...item
      }
    })
    setState(prev => ({
      ...prev,
      GPU_LIST: [...gpuList, ...localItems]
    }));
  }

  const loadNetwork = async () => {
    const netRes: any = await getNetWork()
    const topoList = netRes.network_topology.map((item: any) => {
      return {
        key: item,
        label: item,
        value: item
      }
    })
    setProject({
      curNetwork: {
        network_bandwidth: netRes?.network_bandwidth,
        network_topology: netRes.network_topology.length > 0 ? netRes.network_topology[0] : '',
        ...curNetwork
      }
    });

    setState(prev => ({
      ...prev,
      TOPO_LIST: [...topoList],
      netInfo: {
        network_bandwidth: netRes.network_bandwidth,
        network_topology: netRes.network_topology.length > 0 ? netRes.network_topology[0] : ''
      }
    }))
  }

  const showAddModal = () => {
    setState({
      ...state,
      showAddModal: true
    })
  }
  const closeAddModal = () => {
    setState({
      ...state,
      showAddModal: false
    })
  }
  const setNewGpu = (newItem: any) => {
    setState({
      ...state,
      newGpu: newItem
    })
  }
  const addItemToList = () => {
    const isNotComplete = PARAMS_LIST.find((p => !state.newGpu[p.key]))
    if (isNotComplete) {
      message.warn('Please fill it out completely!')
      return
    }
    const newItem = {
      ...state.newGpu,
      key: state.newGpu.name,
      label: state.newGpu.name,
      value: state.newGpu.name,
    }
    const newGpuList = [...state.GPU_LIST, newItem]
    setState({
      ...state,
      GPU_LIST: newGpuList,
      showAddModal: false
    })
    setProject({
      curGpu: {
        ...newItem
      }
    });
    const localItems = JSON.parse(localStorage.getItem('local_gpus') || '[]') || []
    localStorage.setItem('local_gpus', JSON.stringify([...localItems, newItem]))
  }
  useEffect(() => {
    loadGpuList()
    loadNetwork()
    initCurGpu()
  }, []);


  return (
    <div className={styles.nest}>
      <p className={styles.section_title}>
        {t('select title')} GPU
      </p>
      <div className={styles.section_content}>
        <Select
          options={state.GPU_LIST}
          value={curGpu?.value}
          placeholder={t('Please select')}
          onChange={handleItemClick}
          dropdownRender={(menu) => (
            <>
              {menu}
              {/* <Divider />
              <Button type="link" icon={<PlusOutlined />}
                style={{ padding: '0 10px' }}
                onClick={showAddModal}>
                {t('add item')}
              </Button> */}
            </>
          )}
        >
        </Select>
      </div>
      <p className={styles.section_title}>
        {/* Parameters */}
        {t('parameters')}
      </p>
      <div>
        {curGpu?.value ?
          <div className={styles.gpu_params}>
            {PARAMS_LIST.map((pItem, _idx) =>
              <div key={_idx}>
                <div className={styles.gpu_params_item}>
                  <div className={styles.gpu_params_label}>{pItem.title}</div>
                  <div className={styles.gpu_params_value}>{curGpu[pItem.key].toString()}
                  </div>
                </div>
                {_idx < PARAMS_LIST.length - 1 && <Divider />}
              </div>)}
          </div>
          :
          <div className={styles.to_tips}>
            <Empty />
          </div>
        }
      </div>

      <div className={styles.cluster_param_item}>
        <div className={styles.param_item_label}>Gpu Numbers</div>
        <div className={styles.param_item_value}>
          <InputNumber
            width={100}
            min={1}
            value={curGpu?.num_procs} onChange={(val: any) => {
              setProject({
                curGpu: {
                  ...curGpu,
                  num_procs: val
                }
              });
            }} />
        </div>
      </div>




      <div className={styles.group_slider}>
        {SLIDER_LIST.map((cf: any) => {
          return (
            <div className={styles['group-list-item']} key={cf.key}>
              <div className={styles['item-wrapper']}>
                <span>
                  {cf.title}
                </span>
                <InputNumber
                  precision={cf.precision || 0}
                  width={100}
                  min={cf.min}
                  max={cf.max}
                  value={curNetwork?.[cf.key]}
                  onChange={(val) => {
                    setProject({ curNetwork: { ...curNetwork, [cf.key]: val } });
                  }}
                />
              </div>
              <Slider
                min={cf.min}
                max={cf.max}
                onChange={(val) => {
                  setChangeLog(cf.title, val, curNetwork?.[cf.key])
                  setProject({ curNetwork: { ...curNetwork, [cf.key]: val } });
                }}
                value={curNetwork?.[cf.key]}
                step={cf.step}
              />
            </div>
          );
        })}
      </div>

      <div className={styles.cluster_param_item}>
        <div className={styles.param_item_label}>Network Topology</div>
        <div className={styles.param_item_value}>

          <Select
            options={state.TOPO_LIST}
            value={curNetwork?.network_topology}
            onChange={(val: any) => {
              setProject({
                curNetwork: {
                  ...curNetwork,
                  network_topology: val
                }
              });
            }}
          >
          </Select>
        </div>
      </div>


      <Drawer title={t('add item')} placement="right" width={600}
        // getPopupContainer={(node: any) => {
        //   if (node) {
        //     return node.parentNode;
        //   }
        //   return document.body;
        // }}
        onClose={closeAddModal}
        open={state.showAddModal}>
        <div className="gpu_params">
          {PARAMS_LIST.map((pItem, _idx) =>
            <div key={_idx}>
              <div className="gpu_params_item">
                <div className="gpu_params_label">{pItem.title}</div>
                <div className="gpu_params_value">
                  {pItem.key === 'name'
                    ?
                    <Input
                      required
                      className="number_controls"
                      value={state.newGpu[pItem.key]} onChange={(e: any) => {
                        setNewGpu({
                          ...state.newGpu,
                          [pItem.key]: e.target.value
                        });
                      }} />
                    :
                    <InputNumber controls={false}
                      required
                      className="number_controls"
                      value={state.newGpu[pItem.key]} onChange={(val: any) => {
                        setNewGpu({
                          ...state.newGpu,
                          [pItem.key]: val
                        });
                      }} />}
                </div>
              </div>
              {_idx < PARAMS_LIST.length - 1 && <Divider />}
            </div>)}
        </div>
        <div className='add-item-footer'>
          <Button onClick={closeAddModal}>
            {t('cancel')}
          </Button>
          <Button type="primary" onClick={addItemToList}>
            {t('add')}
          </Button>
        </div>
      </Drawer>
    </div>
  );
};

export default GpuSelection;
