import { FC, useState } from 'react';

import { useReconciliate, useTableDisagreement } from '../core/api';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { useAppContext } from '../core/context';

/*
 * Manage disagreement in annotations
 */

export const AnnotationDisagreementManagement: FC<{
  projectSlug: string;
}> = ({ projectSlug }) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  // available labels from context
  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];

  // get disagreement table
  const { tableDisagreement, users } = useTableDisagreement(projectSlug, currentScheme);
  const { postReconciliate } = useReconciliate(projectSlug, currentScheme || null);

  // state elements to validate
  const [changes, setChanges] = useState<{ [key: string]: string }>({});

  // function to validate changes
  const validateChanges = () => {
    Object.entries(changes).map(([id, label]) => {
      postReconciliate(id, label, users || []);
      setChanges({});
    });
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-3 m-3">
          <button className="btn btn-primary" onClick={validateChanges}>
            Validate changes
          </button>
        </div>
      </div>
      {tableDisagreement?.map((element, index) => (
        <div className="alert alert-warning" role="alert" key={index}>
          <div className="row">
            <div>
              <span className="badge bg-light text-dark">{element.id as string}</span>
              <span>{element.text as string}</span>
            </div>

            {element.annotations && (
              <div className="d-inline-flex align-items-center  mt-2">
                {Object.entries(element.annotations).map(([key, value], _) => (
                  <div key={key}>
                    <span className="badge bg-warning text-dark me-2">
                      {value}
                      <span className="badge rounded-pill bg-light text-dark m-1">{key}</span>
                    </span>
                  </div>
                ))}

                <select
                  className="form-select w-25"
                  onChange={(event) =>
                    setChanges({ ...changes, [element.id as string]: event.target.value })
                  }
                >
                  <option>Select a label to arbitrage</option>
                  {availableLabels.map((e) => (
                    <option key={e}>{e}</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
