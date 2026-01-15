import classNames from 'classnames';
import { ChangeEvent, Dispatch, FC, SetStateAction, useEffect, useMemo, useState } from 'react';
import { FaMapMarkedAlt } from 'react-icons/fa';
import { GiTigerHead } from 'react-icons/gi';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import { MdDisplaySettings } from 'react-icons/md';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';

import { keys, sortBy } from 'lodash';
import { useGetQuickModel, useStatistics } from '../../core/api';
import { useAppContext } from '../../core/context';
import { isValidRegex } from '../../core/utils';
import { AnnotationTagFilterSelect } from './AnnotationTagFilterSelect';

interface AnnotationModeFormProps {
  fetchNextElement: () => void;
  setActiveMenu: Dispatch<SetStateAction<boolean>>;
  setShowDisplayViz: Dispatch<SetStateAction<boolean>>;
  setShowDisplayConfig: Dispatch<SetStateAction<boolean>>;
  nSample: number | null;
  statistics: ReturnType<typeof useStatistics>['statistics'];
}

function optionValue(option: Record<string, string | undefined>) {
  const value = sortBy(keys(option))
    .map((k) => option[k])
    .join('|');
  return value;
}

// define the component to configure selection mode
export const AnnotationModeForm: FC<AnnotationModeFormProps> = ({
  fetchNextElement,
  setActiveMenu,
  setShowDisplayViz,
  setShowDisplayConfig,
  nSample,
  statistics,
}) => {
  const {
    appContext: {
      currentScheme,
      selectionConfig,
      currentProject: project,
      activeModel,
      phase,
      currentProjection,
    },
    setAppContext,
  } = useAppContext();

  // API call to get the current model & refetch
  // TODO : MODEL SELECTION TO CHANGE
  const name = null;
  const { currentModel } = useGetQuickModel(
    project ? project.params.project_slug : null,
    name,
    project,
  );

  const [availableLabels, setAvailableLabels] = useState<string[]>([]);

  const statisticsDataset = useMemo(() => {
    if (phase === 'train') return `${statistics?.train_annotated_n}/${statistics?.train_set_n}`;
    if (phase === 'valid') return `${statistics?.valid_annotated_n}/${statistics?.valid_set_n}`;
    if (phase === 'test') return `${statistics?.test_annotated_n}/${statistics?.test_set_n}`;
    return '';
  }, [phase, statistics]);

  // keep availableLabels up to date
  useEffect(() => {
    // case where the quick model is dichotomize on a specific label
    if (currentModel && currentModel.params && currentModel.params['dichotomize']) {
      setAvailableLabels([
        currentModel.params['dichotomize'] as string,
        'not-' + currentModel.params['dichotomize'],
      ]);
    } else if (currentScheme && project?.schemes.available[currentScheme])
      setAvailableLabels(project.schemes.available[currentScheme].labels);
    else setAvailableLabels([]);
  }, [currentModel, setAvailableLabels, currentScheme, project?.schemes.available]);

  // change dataset : there should be a navigation to reset element id
  const changeDataSet = (e: ChangeEvent<HTMLSelectElement>) => {
    setAppContext((prev) => ({
      ...prev,
      phase: e.target.value,
    }));
  };

  const isValid = project?.params.valid;
  const isTest = project?.params.test;

  const selectionModeOptions: { mode: string; label_maxprob?: string; value: string }[] =
    useMemo(() => {
      const modes = (
        (phase === 'train' && activeModel
          ? project?.next.methods.filter((m) => m !== 'maxprob')
          : project?.next.methods_min) || []
      ).map((mode) => ({ mode, label_maxprob: undefined }));
      const probLabels =
        phase === 'train' && activeModel
          ? availableLabels.map((l) => ({
              mode: 'maxprob',
              label_maxprob: l,
            }))
          : [];
      return [...modes, ...probLabels].map((o) => ({ ...o, value: optionValue(o) }));
    }, [phase, activeModel, project?.next.methods, project?.next.methods_min, availableLabels]);

  return (
    <form className="annotation-mode">
      {/* FIRST row  */}
      <div className="selectors">
        {/* left container: main selectors */}
        <div>
          {/* PHASE - DATASET */}
          <div className="at-input-group">
            <label className=" small-gray">Dataset</label>
            <select
              value={phase}
              onChange={(e) => {
                changeDataSet(e);
              }}
            >
              <option value="train">train</option>
              {isValid && <option value="valid">validation</option>}
              {isTest && <option value="test">test</option>}
            </select>
          </div>
          {/* Active Mode */}
          {phase === 'train' && (
            <div className="at-input-group">
              <label className=" small-gray">Active mode</label>
              <div>
                <button
                  className="button active-mode-button"
                  type="button"
                  onClick={() => setActiveMenu((prev) => !prev)}
                >
                  <GiTigerHead
                    size={30}
                    className="activelearning"
                    style={{ color: activeModel ? 'green' : 'grey', cursor: 'pointer' }}
                    title="Active learning"
                  />
                  <span>{activeModel ? 'active' : 'inactive'}</span>{' '}
                </button>
              </div>
            </div>
          )}

          <div className="at-input-group">
            <label className="small-gray">Selection method</label>
            <Select
              className="react-select"
              options={selectionModeOptions}
              value={selectionModeOptions.find(
                (o) =>
                  o.value ===
                  optionValue({
                    mode: selectionConfig.mode,
                    label_maxProb: selectionConfig.label_maxprob,
                  }),
              )}
              getOptionLabel={(o) =>
                o.mode === 'maxprob' ? `Maxprob on ${o.label_maxprob}` : o.mode
              }
              onChange={(option) => {
                if (option !== null) {
                  setAppContext((prev) => ({
                    ...prev,
                    selectionConfig: {
                      ...selectionConfig,
                      mode: option.mode,
                      label_maxprob: option.label_maxprob,
                    },
                  }));
                }
              }}
            />
          </div>
        </div>
        {/* CONTENT */}
        <div>
          <div className="at-input-group">
            <label className=" small-gray">Filter by Tag/Users</label>
            <AnnotationTagFilterSelect availableLabels={availableLabels} />
          </div>

          {
            // input validated on deselect
          }
          <div className="at-input-group">
            <label htmlFor="select_regex" className=" small-gray">
              Filter by content
              <HiOutlineQuestionMarkCircle id="regex-tooltip" />
            </label>
            <input
              className={classNames(
                'searchhelp',
                selectionConfig.filter && !isValidRegex(selectionConfig.filter) ? 'is-invalid' : '',
              )}
              type="text"
              id="select_regex"
              placeholder="Enter a regex"
              value={selectionConfig.filter}
              onChange={(e) => {
                setAppContext((prev) => ({
                  ...prev,
                  selectionConfig: { ...prev.selectionConfig, filter: e.target.value },
                }));
              }}
            />
            <div className="invalid-feedback">Regex not valid</div>
            <Tooltip anchorSelect="#regex-tooltip">
              Use CONTEXT= or QUERY= for specific requests
            </Tooltip>
          </div>
          {currentProjection && (
            <div>
              {/* LOCK on UMAP */}

              <div className="at-input-group">
                <label className=" small-gray">Filter by Projection</label>
                <div>
                  <button
                    className="button"
                    type="button"
                    onClick={() => setShowDisplayViz((p) => !p)}
                  >
                    <FaMapMarkedAlt
                      size={30}
                      style={{
                        color: selectionConfig.frameSelection ? 'green' : 'grey',
                        cursor: 'pointer',
                      }}
                      title="Map"
                      id="map-icon"
                    />

                    <Tooltip anchorSelect="#map-icon" place="top">
                      Map frame selection
                    </Tooltip>
                  </button>
                  <span className="badge info">
                    {selectionConfig.frameSelection ? 'active' : 'inactive'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="submit-container">
        {/* TODO: find a design for this */}

        <small className="d-flex text-muted text-end flex-column justify-content-between flex-grow-1">
          {statistics ? (
            <>
              <span>Annotated:&nbsp;{statisticsDataset}</span>
              <span>Selection:&nbsp;{nSample || 'na'}</span>
            </>
          ) : (
            'na'
          )}
        </small>

        <button
          type="button"
          className="btn-primary-action"
          onClick={() => {
            fetchNextElement();
          }}
          title="Get next element with the selection mode"
        >
          <LuRefreshCw size={20} />
          <Tooltip anchorSelect=".getelement" place="top">
            Get next element with the selection mode
          </Tooltip>
        </button>

        <button
          type="button"
          className="btn-secondary-action"
          onClick={() => {
            setShowDisplayConfig((p) => !p);
          }}
          title="Display config menu"
        >
          <MdDisplaySettings size={20} />
        </button>
      </div>
    </form>
  );
};
