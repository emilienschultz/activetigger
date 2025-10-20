import { FC, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { useParams } from 'react-router-dom';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/context';
import { ModelDescriptionModel } from '../types';

import { BertModelManagement } from '../components/BertModelManagement';
import { SimpleModelManagement } from '../components/SimpleModelManagement';

/**
 * Component to manage model training
 */

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing, phase },
  } = useAppContext();

  const [activeKey, setActiveKey] = useState<string>('simple');

  // available models
  const availableBertModels = useMemo(() => {
    if (currentScheme && project?.languagemodels?.available?.[currentScheme]) {
      return Object.keys(project.languagemodels.available[currentScheme]);
    }
    return [];
  }, [project, currentScheme]);
  const baseSimpleModels = project?.simplemodel.options ? project?.simplemodel.options : {};

  const availableSimpleModels = useMemo(
    () =>
      project?.simplemodel.available
        ? (project?.simplemodel.available as { [key: string]: ModelDescriptionModel[] })
        : {},
    [project?.simplemodel.available],
  );
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'simple')}
            >
              <Tab eventKey="simple" title="Simple">
                <div className="explanations">Train machine learning models based on features</div>

                <SimpleModelManagement
                  projectName={projectSlug || null}
                  currentScheme={currentScheme || null}
                  baseSimpleModels={
                    baseSimpleModels as unknown as Record<string, Record<string, number>>
                  }
                  availableSimpleModels={availableSimpleModels[currentScheme || ''] || []}
                  availableFeatures={availableFeatures}
                  availableLabels={availableLabels}
                  kindScheme={kindScheme}
                />
              </Tab>
              <Tab eventKey="models" title="BERT" onSelect={() => setActiveKey('models')}>
                <div className="explanations">Train BERT models</div>
                <BertModelManagement
                  projectSlug={projectSlug || null}
                  currentScheme={currentScheme || null}
                  availableBertModels={availableBertModels}
                  isComputing={isComputing}
                  project={project || null}
                />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
