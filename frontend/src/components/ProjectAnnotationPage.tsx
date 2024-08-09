import { range } from 'lodash';
import { ChangeEvent, FC, useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useAddAnnotation, useGetElementById, useGetNextElementId } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { ElementOutModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, reFetchCurrentProject, currentProject: project, selectionConfig },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element

  // be sure to have a scheme selected
  if (!projectName) return null;
  if (!authenticatedUser?.username) return null;
  if (!currentScheme) {
    notify({ type: 'warning', message: 'You need to select first a scheme' });
    navigate(`/projects/${projectName}`);
    return null;
  }

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(projectName, currentScheme);
  const { getElementById } = useGetElementById(projectName, currentScheme);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(
    projectName,
    currentScheme,
    authenticatedUser?.username,
  );

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
    getNextElementId(selectionConfig).then((nextElementId) => {
      if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
    });
  }, [projectName, navigate]);

  useEffect(() => {
    if (elementId === undefined) {
      getNextElementId(selectionConfig).then((nextElementId) => {
        if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
      });
    } else {
      //fetch element information (text and labels)
      getElementById(elementId).then(setElement);
    }
  }, [elementId]);

  return (
    <ProjectPageLayout projectName={projectName} currentAction="annotate">
      <div className="container-fluid">
        <div className="row">
          <h2 className="subsection">Annotation</h2>
          <span className="explanations">Configure selection mode and annotate data</span>
        </div>
        <div className="row">
          <div className="col-6 ">
            <div>
              <label>Selection mode</label>
              <select onChange={handleSelectChangeMode}>
                {availableModes.map((e, i) => (
                  <option key={i}>{e}</option>
                ))}
              </select>
            </div>
            {selectedMode == 'maxprob' && (
              <div>
                <label>Label</label>
                <select onChange={(e) => (selectionConfig.label = e.target.value)}>
                  {availableLabels.map((e, i) => (
                    <option key={i}>{e}</option>
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
              {availableSamples.map((e, i) => (
                <option key={i}>{e}</option>
              ))}{' '}
            </select>
          </div>
        </div>
      </div>
      <hr />

      <div className="row">
        <div className="col-10 annotation-frame my-4">{element?.text}</div>
      </div>
      <div className="row">
        <div className="d-flex flex-wrap gap-2 justify-content-center">
          {availableLabels.map((i) => (
            <button
              key={i}
              value={i}
              className="btn btn-primary grow-1"
              onClick={(e) => {
                if (elementId) {
                  addAnnotation(elementId, e.currentTarget.value).then(navigateToNextElement);
                  // TODO manage erreur
                }
              }}
            >
              {i}
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
      <div>{element ? JSON.stringify(element) : 'loading...'}</div>
    </ProjectPageLayout>
  );
};
