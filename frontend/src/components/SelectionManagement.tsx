import { ChangeEvent, FC, useEffect, useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { Tooltip } from 'react-tooltip';
import { useGetSimpleModel } from '../core/api';
import { useAppContext } from '../core/context';

// define the component to configure selection mode
export const SelectionManagement: FC = () => {
  const {
    appContext: {
      currentScheme,
      selectionConfig,
      currentProject: project,
      activeSimpleModel,
      phase,
    },
    setAppContext,
  } = useAppContext();

  const availableModes =
    activeSimpleModel && project ? project.next.methods : project?.next.methods_min;

  const availableSamples = project?.next.sample ? project?.next.sample : [];

  const availableUsers = project?.users ? project?.users : [];

  // API call to get the current model & refetch
  // TODO : MODEL SELECTION TO CHANGE
  const name = null;
  const { currentModel } = useGetSimpleModel(
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
    // case where the simple model is dichotomize on a specific label
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

  console.log(project);

  return (
    <div className="w-100">
      <div className="d-flex align-items-center">
        {selectionConfig.frameSelection && <FaLock className="m-2" size={20} />}
        <div className="mx-2">
          <label className="form-label label-small-gray">Selection</label>
          <select
            className="form-select"
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
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
              <label className="form-label label-small-gray">
                Maxprob on
                <select
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, label_maxprob: e.target.value },
                    }));
                  }}
                  className="form-select"
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
        <div className="mx-2">
          <label className="form-label label-small-gray">Dataset</label>
          <select className="form-select" value={phase} onChange={changeDataSet}>
            <option value="train">train</option>
            {isValid && <option value="valid">validation</option>}
            {isTest && <option value="test">test</option>}
          </select>
        </div>
        <div className="mx-2 w-25">
          <label className="form-label label-small-gray">Tagged</label>
          <select className="form-select" onChange={changeSample} value={selectionConfig.sample}>
            {availableSamples.map((e, i) => (
              <option key={i}>{e}</option>
            ))}{' '}
          </select>
          {
            // label selection for tagged elemnts
            selectionConfig.sample == 'tagged' && (
              <>
                <label className="form-label label-small-gray">
                  On label
                  <select
                    onChange={(e) => {
                      setAppContext((prev) => ({
                        ...prev,
                        selectionConfig: { ...selectionConfig, label: e.target.value },
                      }));
                    }}
                    className="form-select"
                    value={selectionConfig.label}
                  >
                    {selectionConfig.sample == 'tagged' && <option key="">All</option>}
                    {availableLabels.map((e, i) => (
                      <option key={i}>{e}</option>
                    ))}{' '}
                  </select>
                </label>
                <label htmlFor="select_user" className="form-label label-small-gray">
                  By user
                  <select
                    className="form-select"
                    id="select_user"
                    onChange={(e: ChangeEvent<HTMLSelectElement>) => {
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
              </>
            )
          }
        </div>

        {
          // input validated on deselect
        }
        <div className="w-50">
          <label htmlFor="select_regex" className="form-label label-small-gray">
            Filter
          </label>
          <input
            className="form-control searchhelp"
            type="text"
            id="select_regex"
            placeholder="Enter a regex"
            value={selectionConfig.filter}
            onChange={(e) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: { ...selectionConfig, filter: e.target.value },
              }));
            }}
          />
          <Tooltip anchorSelect=".searchhelp" place="top">
            Use CONTEXT= or QUERY= for specific requests
          </Tooltip>
        </div>
      </div>
    </div>
  );
};
