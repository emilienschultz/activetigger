import classNames from 'classnames';
import { ChangeEvent, Dispatch, FC, SetStateAction, useEffect, useMemo, useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { GiTigerHead } from 'react-icons/gi';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';

import { keys, pickBy, sortBy } from 'lodash';
import { useGetQuickModel } from '../../core/api';
import { useAppContext } from '../../core/context';
import { isValidRegex } from '../../core/utils';

interface AnnotationModeFormProps {
  settingChanged: boolean;
  setSettingChanged: Dispatch<SetStateAction<boolean>>;
  refetchElement: () => void;
  setActiveMenu: Dispatch<SetStateAction<boolean>>;
}

function optionValue(option: Record<string, string | undefined>) {
  return sortBy(keys(option))
    .map((k) => option[k])
    .join('|');
}

// define the component to configure selection mode
export const AnnotationModeForm: FC<AnnotationModeFormProps> = ({
  settingChanged,
  setSettingChanged,
  refetchElement,
  setActiveMenu,
}) => {
  const {
    appContext: { currentScheme, selectionConfig, currentProject: project, activeModel, phase },
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

  const [availableLabels, setAvailableLabels] = useState<string[]>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [],
  );

  // update if new model
  useEffect(() => {
    // case where the quick model is dichotomize on a specific label
    if (currentModel && currentModel.params && currentModel.params['dichotomize']) {
      setAvailableLabels([
        currentModel.params['dichotomize'] as string,
        'not-' + currentModel.params['dichotomize'],
      ]);
    }
  }, [currentModel]);

  // force a default label
  useEffect(() => {
    if (!selectionConfig.label && availableLabels && availableLabels.length > 0) {
      setAppContext((prev) => ({
        ...prev,
        selectionConfig: { ...selectionConfig, label: availableLabels[0] },
      }));
    }
  }, [availableLabels, selectionConfig, setAppContext]);

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
        (activeModel
          ? project?.next.methods.filter((m) => m !== 'maxprob')
          : project?.next.methods_min) || []
      ).map((mode) => ({ mode, label_maxprob: undefined }));
      const probLabels = activeModel
        ? availableLabels.map((l) => ({
            mode: 'maxprob',
            label_maxprob: l,
          }))
        : [];
      return [...modes, ...probLabels].map((o) => ({ ...o, value: optionValue(o) }));
    }, [activeModel, project?.next.methods, project?.next.methods_min, availableLabels]);

  const tagFilterOptions: {
    sample: string | undefined;
    user?: string | undefined;
    label?: string | undefined;
    value: string;
  }[] = useMemo(() => {
    const samples = project?.next.sample.map((s) => ({ sample: s })) || [];
    const labels = availableLabels.map((l) => ({
      sample: 'tagged',
      label: l,
    }));
    const users =
      project?.users?.users.map((u) => ({
        sample: 'tagged',
        user: u,
      })) || [];
    return [...samples, ...labels, ...users].map((o) => ({ ...o, value: optionValue(o) }));
  }, [project?.next.sample, availableLabels, project?.users?.users]);

  console.log(
    tagFilterOptions,
    optionValue({
      sample: selectionConfig.sample,
      label: selectionConfig.label,
      user: selectionConfig.user,
    }),
  );

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
                if (!settingChanged) setSettingChanged(true);
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
                  className="button"
                  type="button"
                  onClick={() => setActiveMenu((prev) => !prev)}
                >
                  <GiTigerHead
                    size={30}
                    className="activelearning"
                    style={{ color: activeModel ? 'green' : 'grey', cursor: 'pointer' }}
                    title="Active learning"
                  />
                  <Tooltip anchorSelect=".activelearning" place="top">
                    Active learning
                  </Tooltip>
                </button>
                <span className="badge info">{activeModel ? 'active' : 'inactive'}</span>
              </div>
            </div>
          )}
        </div>

        {/* SELECTION */}
        <div>
          <div className="at-input-group">
            <label className="small-gray">Select next item by</label>
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
                  if (!settingChanged) setSettingChanged(true);
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
            <label className=" small-gray">Filter by Tag</label>
            <Select
              className="react-select"
              options={tagFilterOptions}
              value={tagFilterOptions.find(
                (o) =>
                  o.value ===
                  optionValue({
                    sample: selectionConfig.sample,
                    label: selectionConfig.label,
                    user: selectionConfig.user,
                  }),
              )}
              getOptionLabel={(o) =>
                o.sample === 'tagged' && o.label !== undefined
                  ? `Tagged as ${o.label}`
                  : o.sample === 'tagged' && o.user !== undefined
                    ? `Tagged by ${o.user}`
                    : o.sample || ''
              }
              onChange={(option) => {
                if (option !== null) {
                  if (!settingChanged) setSettingChanged(true);
                  setAppContext((prev) => ({
                    ...prev,
                    selectionConfig: {
                      ...selectionConfig,
                      ...pickBy(option, (v, k) => v !== undefined && k !== 'value'),
                    },
                  }));
                }
              }}
            />
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
                if (!settingChanged) setSettingChanged(true);
                setAppContext((prev) => ({
                  ...prev,
                  selectionConfig: { ...selectionConfig, filter: e.target.value },
                }));
              }}
            />
            <div className="invalid-feedback">Regex not valid</div>
            <Tooltip anchorSelect="#regex-tooltip">
              Use CONTEXT= or QUERY= for specific requests
            </Tooltip>
          </div>
          <div>
            {selectionConfig.frameSelection && <FaLock className="mx-2 lock" size={20} />}
            <Tooltip anchorSelect=".lock" place="top">
              A frame is locked, go to projection to change
            </Tooltip>
          </div>
        </div>
        <div className="submit-container">
          <button
            type="submit"
            className={classNames(
              'btn-primary-action getelement d-flex align-center',
              settingChanged ? 'setting-changed' : '',
            )}
            onClick={(e) => {
              e.preventDefault();
              refetchElement();
              setSettingChanged(false);
            }}
            title="Get next element with the selection mode"
          >
            <LuRefreshCw size={20} />
            <Tooltip anchorSelect=".getelement" place="top">
              Get next element with the selection mode
            </Tooltip>
          </button>
        </div>
      </div>
    </form>
  );
};
