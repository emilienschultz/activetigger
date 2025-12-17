import { ChangeEvent, Dispatch, FC, SetStateAction, useEffect, useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useGetQuickModel } from '../core/api';
import { useAppContext } from '../core/context';

interface SelectionManagementProps {
  settingChanged: boolean;
  setSettingChanged: Dispatch<SetStateAction<boolean>>;
}

// define the component to configure selection mode
export const SelectionManagement: FC<SelectionManagementProps> = ({
  settingChanged,
  setSettingChanged,
}) => {
  const {
    appContext: { currentScheme, selectionConfig, currentProject: project, activeModel, phase },
    setAppContext,
  } = useAppContext();

  const availableModes = activeModel && project ? project.next.methods : project?.next.methods_min;

  const availableSamples = project?.next.sample ? project?.next.sample : [];

  const availableUsers = project?.users?.users ? project?.users?.users : [];

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

  const changeSample = (e: ChangeEvent<HTMLSelectElement>) => {
    setAppContext((prev) => ({
      ...prev,
      selectionConfig: { ...selectionConfig, sample: e.target.value },
    }));
  };

  const isValid = project?.params.valid;
  const isTest = project?.params.test;

  return (
    // NOTE: Axel: Not much refactor cause more is coming
    <div className="d-flex align-items-center justify-content-between">
      <div>
        <label className="label-small-gray">Dataset</label>
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
      <div>
        <label className="label-small-gray">Selection</label>
        <select
          onChange={(e: ChangeEvent<HTMLSelectElement>) => {
            if (!settingChanged) setSettingChanged(true);
            setAppContext((prev) => ({
              ...prev,
              selectionConfig: { ...selectionConfig, mode: e.target.value },
            }));
          }}
          value={selectionConfig.mode}
        >
          {(availableModes || []).map((e, i) => (
            <option key={i}>{e}</option>
          ))}
        </select>
        {
          // label selection for maxprob
          selectionConfig.mode == 'maxprob' && (
            <label className="label-small-gray">
              Maxprob on
              <select
                onChange={(e) => {
                  if (!settingChanged) setSettingChanged(true);
                  setAppContext((prev) => ({
                    ...prev,
                    selectionConfig: { ...selectionConfig, label_maxprob: e.target.value },
                  }));
                }}
                value={selectionConfig.label_maxprob}
              >
                {availableLabels.map((e, i) => (
                  <option key={i}>{e}</option>
                ))}{' '}
              </select>
            </label>
          )
        }
      </div>

      <div className="parameter-div">
        <label className="label-small-gray">Tagged</label>
        <select
          onChange={(e) => {
            if (!settingChanged) setSettingChanged(true);
            changeSample(e);
          }}
          value={selectionConfig.sample}
        >
          {availableSamples.map((e, i) => (
            <option key={i}>{e}</option>
          ))}{' '}
        </select>
      </div>
      {
        // label selection for tagged elemnts
        selectionConfig.sample == 'tagged' && (
          <>
            <div className="parameter-div">
              <label className="label-small-gray">
                On label
                <select
                  onChange={(e) => {
                    if (!settingChanged) setSettingChanged(true);
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, label: e.target.value },
                    }));
                  }}
                  value={selectionConfig.label}
                >
                  {selectionConfig.sample == 'tagged' && <option key="">All</option>}
                  {availableLabels.map((e, i) => (
                    <option key={i}>{e}</option>
                  ))}{' '}
                </select>
              </label>
            </div>
            <div className="parameter-div">
              <label htmlFor="select_user" className="label-small-gray">
                By user
                <select
                  id="select_user"
                  onChange={(e: ChangeEvent<HTMLSelectElement>) => {
                    if (!settingChanged) setSettingChanged(true);
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, user: e.target.value },
                    }));
                  }}
                  value={selectionConfig.user}
                >
                  <option key={null} value={''}>
                    All
                  </option>
                  {availableUsers.map((e, i) => (
                    <option key={i}>{e}</option>
                  ))}
                </select>
              </label>{' '}
            </div>
          </>
        )
      }

      {
        // input validated on deselect
      }
      <div id="regex" className="parameter-div">
        <label htmlFor="select_regex" className="label-small-gray">
          Filter
          <HiOutlineQuestionMarkCircle id="regex-tooltip" />
        </label>
        <input
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
        <Tooltip anchorSelect="#regex-tooltip" style={{ zIndex: '99' }}>
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
  );
};
