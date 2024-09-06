import { useDebounce } from '@uidotdev/usehooks';
import { ChangeEvent, FC, useEffect, useMemo, useState } from 'react';
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

  const currentModel = useMemo(() => {
    return authenticatedUser &&
      currentScheme &&
      project?.simplemodel.available[authenticatedUser?.username]?.[currentScheme]
      ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
      : null;
  }, [project, currentScheme, authenticatedUser]);

  // manage regex input with a debounce
  const [filter, setFilter] = useState<string | null>(null);
  const debouncedFilter = useDebounce(filter, 1000);
  useEffect(() => {
    if (debouncedFilter && debouncedFilter != selectionConfig.filter)
      setAppContext((prev) => ({
        ...prev,
        selectionConfig: { ...selectionConfig, filter: debouncedFilter },
      }));
  }, [debouncedFilter, selectionConfig, setAppContext]);

  return phase == 'test' ? (
    <div>Test mode activated - deactivate first before annotating train set</div>
  ) : (
    <div>
      <div className="explanations">
        Current model : {currentModel ? currentModel['model'] : 'No model trained'}
      </div>
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
          value={selectionConfig.mode}
        >
          {availableModes.map((e, i) => (
            <option key={i}>{e}</option>
          ))}
        </select>
      </div>
      {selectionConfig.mode == 'maxprob' && (
        <div className="d-flex align-items-center justify-content-between">
          <label>Label</label>
          <select
            onChange={(e) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: { ...selectionConfig, label: e.target.value },
              }));
            }}
            className="form-select w-50"
            value={selectionConfig.label}
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
          value={selectionConfig.sample}
        >
          {availableSamples.map((e, i) => (
            <option key={i}>{e}</option>
          ))}{' '}
        </select>
      </div>
      <div className="d-flex align-items-center justify-content-between">
        <label htmlFor="select_regex">Filter</label>
        <input
          className="form-control w-50"
          type="text"
          id="select_regex"
          placeholder="Enter a regex / CONTEXT= for context"
          onChange={(e) => {
            setFilter(e.target.value);
          }}
          value={filter || ''}
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
        Use projection frame to limit element selection
      </label>
    </div>
  );
};
