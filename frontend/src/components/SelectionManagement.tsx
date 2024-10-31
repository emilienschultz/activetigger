import { ChangeEvent, FC, useMemo } from 'react';
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
  }, [project, currentScheme]);

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
  // if (!selectionConfig.label) {
  //   setAppContext((prev) => ({
  //     ...prev,
  //     selectionConfig: { ...selectionConfig, label: availableLabels[0] || '' },
  //   }));
  // }

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
      <div className="d-flex align-items-center">
        <details className="mx-2">
          <summary className="explanations">
            Active selection : {currentModel ? currentModel['model'] : 'No model trained'}
          </summary>
          {currentModel && (
            <div>
              Model parameters :
              <ul>
                {currentModel.params &&
                  (currentModel.params && Object.entries(currentModel.params)).map(
                    ([key, value], i) => (
                      <li key={i}>
                        {key} - {value}
                      </li>
                    ),
                  )}
              </ul>
              Statistics:
              <ul>
                {currentModel.params &&
                  (currentModel.statistics && Object.entries(currentModel.statistics)).map(
                    ([key, value], i) => (
                      <li key={i}>
                        {key} - {JSON.stringify(value)}
                      </li>
                    ),
                  )}
              </ul>
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
                console.log(selectionConfig.frameSelection);
              }}
            />
            Use visualisation frame to lock the selection
          </label>
        </details>
      </div>
    </div>
  );
};
