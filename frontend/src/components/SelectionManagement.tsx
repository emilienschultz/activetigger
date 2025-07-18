import { ChangeEvent, FC, useEffect, useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { useGetSimpleModel } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';

// define the component to configure selection mode
export const SelectionManagement: FC = () => {
  const { authenticatedUser } = useAuth();
  const {
    appContext: { currentScheme, selectionConfig, currentProject: project },
    setAppContext,
  } = useAppContext();

  const availableModes =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser.username]?.[currentScheme]
      ? project.next.methods
      : project?.next.methods_min
        ? project?.next.methods_min
        : [];

  const availableSamples = project?.next.sample ? project?.next.sample : [];

  const availableUsers = project?.users ? project?.users : [];

  // API call to get the current model & refetch
  const { currentModel } = useGetSimpleModel(
    project ? project.params.project_slug : null,
    currentScheme || null,
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

  return (
    <div className="w-100">
      <div className="d-flex align-items-center">
        {selectionConfig.frameSelection && <FaLock className="m-2" size={20} />}
        <div className="mx-2 w-25">
          <label>Sample</label>
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
          {
            // label selection for tagged elemnts
            selectionConfig.sample == 'tagged' && (
              <>
                <label>
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
                <label htmlFor="select_user">
                  By user
                  <select
                    className="form-select mx-2"
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

        <div className="mx-2 w-25">
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
          {
            // label selection for maxprob
            selectionConfig.mode == 'maxprob' && (
              <label>
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

        {
          // input validated on deselect
        }
        <div className="w-50">
          <label htmlFor="select_regex">Filter</label>
          <input
            className="form-control"
            type="text"
            id="select_regex"
            placeholder="Search / Regex / CONTEXT= / QUERY="
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
    </div>
  );
};
