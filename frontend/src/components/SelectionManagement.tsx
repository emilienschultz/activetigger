import { ChangeEvent, FC, useEffect, useMemo } from 'react';
import { FcStatistics } from 'react-icons/fc';
import { useGetSimpleModel } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';

// define the component to configure selection mode
export const SelectionManagement: FC = () => {
  const { authenticatedUser } = useAuth();
  const {
    appContext: { currentScheme, selectionConfig, currentProject: project, phase },
    setAppContext,
  } = useAppContext();

  const availableLabels = useMemo(() => {
    return currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  }, [currentScheme, project]);

  const availableModes =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser.username]?.[currentScheme]
      ? project.next.methods
      : project?.next.methods_min
        ? project?.next.methods_min
        : [];

  const availableSamples = project?.next.sample ? project?.next.sample : [];

  // const currentModel = useMemo(() => {
  //   return authenticatedUser &&
  //     currentScheme &&
  //     project?.simplemodel.available[authenticatedUser?.username]?.[currentScheme]
  //     ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
  //     : null;
  // }, [project, currentScheme, authenticatedUser]);

  // API call to get the current model & refetch
  const { currentModel } = useGetSimpleModel(
    project ? project.params.project_slug : null,
    currentScheme || null,
    project,
  );

  // useEffect(() => {
  //   reFetchSimpleModel();
  // }, [reFetchSimpleModel, project]);

  // force a default label
  useEffect(() => {
    if (!selectionConfig.label && availableLabels && availableLabels.length > 0) {
      setAppContext((prev) => ({
        ...prev,
        selectionConfig: { ...selectionConfig, label: availableLabels[0] },
      }));
    }
  }, [availableLabels, selectionConfig, setAppContext]);

  return phase == 'test' ? (
    <div>Test mode activated - deactivate first before annotating train set</div>
  ) : (
    <div className="w-100">
      <div className="d-flex align-items-center">
        <div className="mx-2">
          <label>Selection</label>
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
            {availableModes.map((e, i) => (
              <option key={i}>{e}</option>
            ))}
          </select>
        </div>
        {selectionConfig.mode == 'maxprob' && (
          <div className="mx-2 w-25">
            <label>Label</label>
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
              {availableLabels.map((e, i) => (
                <option key={i}>{e}</option>
              ))}{' '}
            </select>
          </div>
        )}
        <div className="mx-2 w-25">
          <label>On</label>
          <select
            className="form-select"
            onChange={(e) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: { ...selectionConfig, sample: e.target.value },
              }));
            }}
            value={selectionConfig.sample}
          >
            {availableSamples.map((e, i) => (
              <option key={i}>{e}</option>
            ))}{' '}
          </select>
        </div>
        {
          // input validated on deselect
        }
        <div className="w-50">
          <label htmlFor="select_regex">Filter</label>
          <input
            className="form-control"
            type="text"
            id="select_regex"
            placeholder="Enter a regex / CONTEXT= for context"
            value={selectionConfig.filter}
            onChange={(e) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: { ...selectionConfig, filter: e.target.value },
              }));
            }}
          />
        </div>
      </div>
      <div className="d-flex align-items-top">
        <details className="mx-2">
          <summary className="explanations">
            Active model <FcStatistics />{' '}
            <span className="badge bg-light text-dark">
              {currentModel ? currentModel['model'] : 'No model trained'}
            </span>
          </summary>
          {currentModel && (
            <div>
              <table className="table table-striped table-hover">
                <thead>
                  <tr>
                    <th>Model parameters</th>
                    <th></th>
                  </tr>
                </thead>
                {currentModel.params &&
                  (currentModel.params && Object.entries(currentModel.params)).map(
                    ([key, value], i) => (
                      <tr key={i}>
                        <td>{key}</td>
                        <td>{value}</td>
                      </tr>
                    ),
                  )}
                <thead>
                  <tr>
                    <th>Indicators</th>
                    <th></th>
                  </tr>
                </thead>
                {currentModel.params &&
                  (currentModel.statistics && Object.entries(currentModel.statistics)).map(
                    ([key, value], i) => (
                      <tr key={i}>
                        <td>{key}</td>
                        <td> {JSON.stringify(value)}</td>
                      </tr>
                    ),
                  )}
              </table>
            </div>
          )}
        </details>
        <details className="mx-2">
          <summary className="explanations">Advanced options</summary>
          <label className="mx-4" style={{ display: 'block' }}>
            <input
              type="checkbox"
              checked={selectionConfig.frameSelection}
              className="mx-2"
              onChange={(_) => {
                setAppContext((prev) => ({
                  ...prev,
                  selectionConfig: {
                    ...selectionConfig,
                    frameSelection: !selectionConfig.frameSelection,
                  },
                }));
                // console.log(selectionConfig.frameSelection);
              }}
            />
            Use visualisation frame to lock the selection
          </label>
        </details>
      </div>
    </div>
  );
};
