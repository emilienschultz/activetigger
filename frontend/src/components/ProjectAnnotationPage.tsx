import { range } from 'lodash';
import { ChangeEvent, FC, useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: { currentScheme, reFetchCurrentProject, currentProject: project, selectionConfig },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();

  // define parameters of the menu
  const availableSamples = project?.next.sample ? project?.next.sample : [];
  const availableLabels = currentScheme && project ? project?.schemes.available[currentScheme] : [];
  // available methods depend if there is a simple model trained for the user/scheme
  // TO TEST, and in the future change the API if possible
  var availableModes = project?.next.methods_min ? project?.next.methods_min : [];
  if (
    project?.simplemodel.available &&
    authenticatedUser &&
    Array(project?.simplemodel.available).includes(authenticatedUser.username) &&
    Array(project?.simplemodel.available[authenticatedUser.username]).includes(currentScheme)
  )
    availableModes = project?.next.methods;

  // manage the hide/visible menu for the label
  const [selectedMode, setSelectedMode] = useState('');
  const handleSelectChangeMode = (e: ChangeEvent<HTMLSelectElement>) => {
    selectionConfig.mode = e.target.value;
    setSelectedMode(e.target.value);
  };

  // update the selection config when the user change a menu
  useEffect(() => {
    setAppContext((prev) => ({ ...prev, selectionConfig: selectionConfig }));
    console.log('Update selectionConfig');
  }, [selectionConfig]);

  const navigateToNextElement = useCallback(async () => {
    // change url using the new elementId
    // const newElementId = await  apiCall()
    //navigate('/projects/${projectName}/annotate/newid');
  }, [projectName, navigate]);

  useEffect(() => {
    if (elementId === undefined) {
      // fetch next elementId
      // change url using the new elementId
      // navigate("/project/${projectName}/annotate/newid");
      //navigateToNextElement();
    } else {
      //fetch element information (text and labels)
    }
  }, [elementId]);

  // we must get the project annotation payload / element
  if (!projectName) return null;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="annotate">
      <div className="container-fluid">
        <div>{JSON.stringify(selectionConfig)}</div>
        <div className="row">
          <h2 className="subsection">Annotation</h2>
        </div>
        <div className="row">
          <div className="col-6">
            <div>
              <label>Selection mode</label>
              <select onChange={handleSelectChangeMode}>
                {availableModes.map((e) => (
                  <option>{e}</option>
                ))}
              </select>
            </div>
            {selectedMode == 'random' && (
              <div>
                <label>Label</label>
                <select onChange={(e) => (selectionConfig.label = e.target.value)}>
                  {availableLabels.map((e) => (
                    <option>{e}</option>
                  ))}{' '}
                </select>
              </div>
            )}
          </div>
        </div>
        <div className="row">
          <div className="col-6">
            <label>On</label>
            <select onChange={(e) => (selectionConfig.sample = e.target.value)}>
              {availableSamples.map((e) => (
                <option>{e}</option>
              ))}{' '}
            </select>
          </div>
        </div>
      </div>
      <div className="row">
        <h1>Du texte à annoter</h1>
      </div>
      <div className="row">
        <h2>labels</h2>
        <div className="d-flex flex-wrap gap-2">
          {range(10).map((i) => (
            <button
              key={i}
              className="btn btn-primary grow-1"
              onClick={() => {
                // add tag to element
                // if pas d'erreur
                navigateToNextElement();
              }}
            >
              blabla {i}
            </button>
          ))}
        </div>
      </div>
      <hr />
      <details>
        <summary>
          <h2>Label management</h2>
        </summary>
        Plein de bordel
      </details>
    </ProjectPageLayout>
  );
};
