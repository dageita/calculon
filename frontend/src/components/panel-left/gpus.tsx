import { FC, useEffect } from 'react';
import { useImmer } from 'use-immer';
import { Select, Divider, Input, InputNumber, Button, Drawer, message } from 'antd'
import Empty from '../empty';
import useModel from 'flooks';
import { getGpuList, getNetWork } from '@/services'
import ProjectModel from '@/models/projectModel';
import styles from './index.less';
import LogModel from '@/models/logModel';
import { useTranslation } from 'react-i18next';

// Bandwidth defaults come from systems/<gpu>.json, but may be overridden for
// the current calculation without modifying the catalog file.
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
    title: 'Intra-node Bandwidth(GB/s)',
    key: 'bus_bandwidth'
  },
  {
    title: 'Inter-node Bandwidth(GB/s)',
    key: 'network_bandwidth'
  },
  {
    title: 'PCIe Bandwidth(GB/s)',
    key: 'pcie_bandwidth'
  }, {
    title: 'support p2p',
    key: 'support_p2p'
  }
]
const EDITABLE_BANDWIDTH_KEYS = new Set(['bus_bandwidth', 'network_bandwidth'])

export interface IGPUSelectionProps { }
const GpuSelection: FC<IGPUSelectionProps> = (props) => {
  const { setProject, curGpu, curNetwork,curCouHasChanged } = useModel(ProjectModel);
  const { setChangeLog } = useModel(LogModel);
  const { t } = useTranslation();

  const handleItemClick = (key: string, item: any) => {
    setChangeLog('GPU', item?.name, curGpu?.name)
    setProject({
      curGpu: {
        ...item,
        num_procs: curGpu?.num_procs
      },
      // Keep topology; BW always comes from the selected GPU / systems JSON.
      curNetwork: {
        ...curNetwork,
        network_bandwidth: item?.network_bandwidth,
      }
    });
  };

  const handleBandwidthChange = (key: string, value: number | null) => {
    const projectUpdate: any = {
      curGpu: {
        ...curGpu,
        [key]: value,
      },
    }
    if (key === 'network_bandwidth') {
      projectUpdate.curNetwork = {
        ...curNetwork,
        network_bandwidth: value,
      }
    }
    setProject(projectUpdate)
  }


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
    if (gpuRes.error) return
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
    // If a GPU is already selected, refresh BW from catalog (systems JSON).
    if (curGpu?.name) {
      const matched = gpuList.find((g: any) => g.name === curGpu.name)
      if (matched) {
        setProject({
          curGpu: {
            ...curGpu,
            ...matched,
            num_procs: curGpu?.num_procs ? curGpu?.num_procs : 1,
          },
          curNetwork: {
            ...curNetwork,
            network_bandwidth: matched.network_bandwidth,
          }
        })
      }
    }
  }

  const loadNetwork = async () => {
    const netRes: any = await getNetWork()
    if (netRes.error) return
    const topologies = netRes?.network_topology || []
    const topoList = topologies.map((item: any) => {
      return {
        key: item,
        label: item,
        value: item
      }
    })
    setProject({
      curNetwork: {
        ...curNetwork,
        network_topology: curNetwork?.network_topology
          || (topologies.length > 0 ? topologies[0] : ''),
        // Prefer GPU systems JSON BW when available.
        network_bandwidth: curGpu?.network_bandwidth ?? curNetwork?.network_bandwidth,
      }
    });

    setState(prev => ({
      ...prev,
      TOPO_LIST: [...topoList],
      netInfo: {
        network_topology: topologies.length > 0 ? topologies[0] : ''
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
    const isNotComplete = PARAMS_LIST.find((p => !state.newGpu[p.key] && state.newGpu[p.key] !== false && state.newGpu[p.key] !== 0))
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
                  <div className={styles.gpu_params_value}>
                    {EDITABLE_BANDWIDTH_KEYS.has(pItem.key)
                      ? <InputNumber
                          controls={false}
                          min={0}
                          step={1}
                          style={{ width: '100%' }}
                          value={curGpu[pItem.key]}
                          onChange={(value) => handleBandwidthChange(pItem.key, value)}
                        />
                      : (curGpu[pItem.key]?.toString?.() ?? String(curGpu[pItem.key]))
                    }
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
      {curGpu?.value && (
        <div className={styles.to_tips}>
          Bandwidth overrides apply to the current configuration only and do not modify the systems JSON.
        </div>
      )}

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
                  network_topology: val,
                  network_bandwidth: curGpu?.network_bandwidth,
                }
              });
            }}
          >
          </Select>
        </div>
      </div>


      <Drawer title={t('add item')} placement="right" width={600}
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
