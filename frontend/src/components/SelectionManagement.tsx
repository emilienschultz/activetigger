import { ChangeEvent, FC, useMemo } from 'react';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';

// define the component to configure selection mode
export const SelectionManagement: FC = () => {
  const {
    appContext: { currentScheme, selectionConfig, currentProject: project },
    setAppContext,
  } = useAppContext();

  const availableLabels = useMemo(() => {
    return currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  }, [project, currentScheme]);

  const { authenticatedUser } = useAuth();

  const availableModes =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser.username]?.[currentScheme]
      ? project.next.methods
      : project?.next.methods_min
        ? project?.next.methods_min
        : [];

  const availableSamples = project?.next.sample ? project?.next.sample : [];

  const currentModel = useMemo(() => {
    return authenticatedUser &&
      currentScheme &&
      project?.simplemodel.available[authenticatedUser?.username]?.[currentScheme]
      ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
      : null;
  }, [project, currentScheme, authenticatedUser]);

  return (
    <div>
      <div className="d-flex align-items-center justify-content-between">
        <label>Selection mode</label>
        <select
          className="form-select w-50"
          onChange={(e: ChangeEvent<HTMLSelectElement>) => {
            setAppContext((prev) => ({
              ...prev,
              selectionConfig: { ...selectionConfig, mode: e.target.value },
            }));
          }}
        >
          {availableModes.map((e, i) => (
            <option key={i}>{e}</option>
          ))}
        </select>
      </div>
      {selectionConfig.mode == 'maxprob' && (
        <div>
          <label>Label</label>
          <select
            onChange={(e) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: { ...selectionConfig, label: e.target.value },
              }));
            }}
          >
            {availableLabels.map((e, i) => (
              <option key={i}>{e}</option>
            ))}{' '}
          </select>
        </div>
      )}
      <div className="d-flex align-items-center justify-content-between">
        <label>On</label>
        <select
          className="form-select w-50"
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              selectionConfig: { ...selectionConfig, sample: e.target.value },
            }));
          }}
        >
          {availableSamples.map((e, i) => (
            <option key={i}>{e}</option>
          ))}{' '}
        </select>
      </div>
      <div className="d-flex align-items-center justify-content-between">
        <label htmlFor="select_regex">Filter</label>
        <input
          className="form-control w-75"
          type="text"
          id="select_regex"
          placeholder="Enter a regex / CONTEXT= for context"
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              selectionConfig: { ...selectionConfig, filter: e.target.value },
            }));
          }}
        />
      </div>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        <input
          type="checkbox"
          checked={selectionConfig.frameSelection}
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
          style={{ marginRight: '10px' }}
        />
        Use zoom frame to select elements
      </label>
      <div>Current model : {currentModel ? currentModel['model'] : 'No model trained'}</div>
    </div>
  );
};
